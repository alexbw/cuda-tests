import os
import numpy as np
import time
from pycuda import driver, compiler, gpuarray, tools
from pycuda.compiler import SourceModule
from pycuda.driver import func_cache
from pycuda.driver import device_attribute
import pycuda.autoinit

class MousePoser(object):
    """A class managing the generative model.
    It's designed to take in joint angles,
    and provide either a likelihood against a real image,
    or a pose of the mouse.

    Behind the scenes, it's doing everything on the graphics card,
    and is managing all of the setup and teardown required.
    """
    def __init__(self, mouseModel=None, maxNumBlocks=30, imageSize=(64,64)):
        super(MousePoser, self).__init__()
        
        self.mouseModel = mouseModel

        self._setup_contexts()
        
        # SET TUNABLE PARAMETERS
        self.maxNumBlocks = maxNumBlocks
        self.maxNumThreads = self.devices[0].get_attribute(device_attribute.MAX_THREADS_PER_BLOCK)
        self.numMicePerPass = self.maxNumBlocks*self.maxNumThreads
        self.resolutionX = imageSize[1]
        self.resolutionY = imageSize[0]
        self.numJoints = self.mouseModel.num_joints

        self._set_kernel_run_sizes(numBlocks=10, numThreads=256)
        self._set_cache_preference()
        self._setup_kernels()
        self._setup_arrays()
        
        

    def _setup_contexts(self):
        # Grab a context for each GPU
        dev = driver.Device(0)
        ctx = dev.make_context()
        self.devices = []
        self.contexts = []
        self.devices.append(dev)
        self.contexts.append(ctx)

        # Get the number of GPUs in this computer
        self.numGPUs = driver.Device(0).count()

        # Grab a context for each GPU
        for i in range(1,self.numGPUs):
            dev = driver.Device(i)
            ctx = dev.make_context()
            self.devices.append(dev)
            self.contexts.append(ctx)
            
    def _set_cache_preference(self, cache_config=driver.func_cache.PREFER_L1):
        # Cache rules everything around me
        for ctx in self.contexts:
            ctx.push()
            ctx.set_cache_config(cache_config)
            ctx.pop()

    def _setup_arrays(self):
        # Data, for each context
        self.synthPixels_gpu = []
        self.realPixels_gpu = []
        self.mouseVertices_gpu = []
        self.mouseVertexIdx_gpu = []
        self.skinnedVertices_gpu = []
        self.jointWeights_gpu = []
        self.jointWeightIndices_gpu = []
        self.jointTransforms_gpu = []
        self.likelihoods_gpu = []
        self.jointRotations_gpu = []
        self.baseJointRotations_gpu = []
        self.jointTranslations_gpu = []

        for ctx in self.contexts:
            ctx.push()
            # Synthetic pixels
            self.synthPixels_cpu = np.zeros((self.resolutionX, self.resolutionY), dtype='float32')
            self.synthPixels_cpu = np.tile(self.synthPixels_cpu, (self.numMicePerPass,1))
            self.synthPixels_gpu.append(gpuarray.to_gpu_async(self.synthPixels_cpu))

            # Real mouse pixels
            self.realPixels_cpu = np.zeros((int(self.resolutionX), int(self.resolutionY)), dtype='float32')
            self.realPixels_cpu += 10*np.random.random(self.realPixels_cpu.shape) # testing only
            self.realPixels_gpu.append(gpuarray.to_gpu_async(self.realPixels_cpu))

            # Mouse vertices
            self.mouseVertices_cpu = self.mouseModel.vertices[:,:3].astype('float32')
            self.mouseVertices_gpu.append(gpuarray.to_gpu_async(self.mouseVertices_cpu))

            # Triangle face indices
            self.mouseVertexIdx_cpu = self.mouseModel.vertex_idx.astype('uint16')
            self.mouseVertexIdx_gpu.append(gpuarray.to_gpu_async(self.mouseVertexIdx_cpu))

            # Skinned vertices
            self.skinnedVertices_cpu = self.mouseVertices_cpu.copy()
            self.skinnedVertices_cpu = np.tile(self.skinnedVertices_cpu, (self.numMicePerPass,1))
            self.skinnedVertices_gpu.append(gpuarray.to_gpu_async(self.skinnedVertices_cpu))

            # Joint weights
            self.jointWeights_cpu = self.mouseModel.nonzero_joint_weights.astype('float32')
            self.jointWeights_gpu.append(gpuarray.to_gpu_async(self.jointWeights_cpu))

            # Joint weight indices
            self.jointWeightIndices_cpu = self.mouseModel.joint_idx.astype('uint16')
            self.jointWeightIndices_gpu.append(gpuarray.to_gpu_async(self.jointWeightIndices_cpu))

            # Okay, for joint transforms,
            # -- [x,y,z]
            # x : left and right sweeps
            # y : twisting
            # z : up and down rears
            
            # Joint transforms
            self.jointTransforms_cpu = np.eye(4, dtype='float32')
            self.jointTransforms_cpu = np.tile(self.jointTransforms_cpu, (self.numMicePerPass*self.numJoints,1))
            self.jointTransforms_gpu.append(gpuarray.to_gpu_async(self.jointTransforms_cpu))

            # Likelihoods
            self.likelihoods_cpu = np.zeros((self.numMicePerPass,), dtype='float32')
            self.likelihoods_gpu.append(gpuarray.to_gpu_async(self.likelihoods_cpu))

            # Joint rotations
            self.jointRotations_cpu = self.mouseModel.joint_rotations.astype('float32')
            self.jointRotations_cpu = np.tile(self.jointRotations_cpu, (self.numMicePerPass,1))
            self.jointRotations_gpu.append(gpuarray.to_gpu_async(self.jointRotations_cpu))

            # Base joint rotations
            self.baseJointRotations_cpu = self.mouseModel.joint_rotations.astype('float32')
            self.baseJointRotations_gpu.append(gpuarray.to_gpu_async(self.baseJointRotations_cpu))

            # Joint translations (we never propose over these)
            self.jointTranslations_cpu = self.mouseModel.joint_translations.astype('float32')
            self.jointTranslations_gpu.append(gpuarray.to_gpu_async(self.jointTranslations_cpu))

            ctx.synchronize()
            ctx.pop()

    def _setup_kernels(self):
        # First, read the code in and template it
        self._read_kernels()

        # Kernels, for each context
        self.raster = []
        self.likelihood = []
        self.fk = []
        self.skinning = []

        basePath = os.path.split(os.path.realpath(__file__))[0]
        includePath = os.path.join(basePath, "include")

        # Now, for each context, compile the kernel
        for ctx in self.contexts:
            ctx.push()
            mod = compiler.SourceModule(self._kernel_code, options=\
                                    ['-I%s' % includePath, \
                                    '--compiler-options', '-w',
                                    '--optimize', '3', \
                                    ], no_extern_c=True)
            self.raster.append(mod.get_function("rasterizeSerial"))
            self.likelihood.append(mod.get_function("likelihoodSerial"))
            self.fk.append(mod.get_function("FKSerial"))
            self.skinning.append(mod.get_function("skinningSerial"))
            ctx.pop()

    def _set_kernel_run_sizes(self, numBlocks=10, numThreads=256):
        """For now, don't worry about auto-tuning specific functions. Everybody
        gets the same block size so that we can do equal work easily.

        The next thing to try is to have integral ratios of block and thread
        differences, and we'll add for loops around each kernel call."""

        assert np.mod(self.maxNumThreads, numThreads) == 0, "Threads must be an integral multiple of max possible"
        assert np.mod(self.maxNumBlocks, numBlocks) == 0, "Blocks must be an integral multiple of max possible"

        self.numBlocksFK,self.numThreadsFK = numBlocks,numThreads
        self.numMiceFK = self.numBlocksFK*self.numThreadsFK
        self.numBlocksRS,self.numThreadsRS = numBlocks,numThreads
        self.numMiceRS = self.numBlocksRS*self.numThreadsRS
        self.numBlocksSK,self.numThreadsSK = numBlocks,numThreads
        self.numMiceSK = self.numBlocksSK*self.numThreadsSK
        self.numBlocksLK,self.numThreadsLK = numBlocks,numThreads
        self.numMiceLK = self.numBlocksLK*self.numThreadsLK
        self.numMice = min([self.numMiceFK, self.numMiceRS, self.numMiceSK, self.numMiceLK])
        self.numMicePerPass = self.numMice

    def _read_kernels(self):
        # Go ahead and grab the kernel code
        with open("raster_test.cu") as kernel_file:
            self._kernel_code_template = kernel_file.read()
        # In this kernel, currently no formatting
        self._kernel_code = self._kernel_code_template.format(resx=self.resolutionX, 
                                                                resy=self.resolutionY, 
                                                                njoints=self.numJoints)




    def get_likelihoods(self, joint_angles, real_mouse_image):

        # TODO: divvy up the joint angles properly
        # TODO: don't use lists, use nparrays and think about indexing you idiot
        # TODO: make sure the right number of calls are being made. This will require asserts.

        numProposals, numAngles = joint_angles.shape
        numProposals /= self.numJoints
        assert numAngles == 3, "Need 3 angles, bro"
        assert np.mod(numProposals, self.numMice) == 0, "Num proposals must be a multiple of %d" % self.numMice

        niter = numProposals/(self.numMice*self.numGPUs)
        
        likelihoods = []
        for this_iter in range(niter):
            idx_iter = this_iter*self.numMice*self.numGPUs

            for i,ctx in enumerate(self.contexts):
                idx_gpu = idx_iter + i*self.numMice
                these_angles = joint_angles[idx_gpu:idx_gpu+self.numMice*self.numJoints,:]
                ctx.push()
                self.jointRotations_gpu[i].set(these_angles)
                
                # Only upload the mouse image on the first iteration
                if this_iter == 0:
                    self.realPixels_gpu[i].set(real_mouse_image)

                #fk
                self.fk[i](self.baseJointRotations_gpu[i],
                        self.jointRotations_gpu[i],
                        self.jointTranslations_gpu[i],
                        self.jointTransforms_gpu[i],
                        grid=(self.numBlocksFK,1,1),
                        block=(self.numThreadsFK,1,1))

                #skin
                # PLEASE ADD THE ABILITY TO ADD SCALING
                self.skinning[i](self.jointTransforms_gpu[i],
                        self.mouseVertices_gpu[i],
                        self.jointWeights_gpu[i],
                        self.jointWeightIndices_gpu[i],
                        self.skinnedVertices_gpu[i],
                        grid=(self.numBlocksSK,1,1),
                        block=(self.numThreadsSK,1,1))

                #raster
                self.raster[i]( self.skinnedVertices_gpu[i], 
                        self.mouseVertices_gpu[i],
                        self.mouseVertexIdx_gpu[i],
                        self.synthPixels_gpu[i],
                        grid=(self.numBlocksRS,1,1),
                        block=(self.numThreadsRS,1,1))

                #likelihood
                self.likelihood[i](self.synthPixels_gpu[i],
                        self.realPixels_gpu[i],
                        self.likelihoods_gpu[i],
                        grid=(self.numBlocksLK,1,1),
                        block=(self.numThreadsLK,1,1))

                likelihoods.append(self.likelihoods_gpu[i].get())
                ctx.pop()

        return np.vstack(likelihoods)


    def get_posed_mice(self, joint_angles):
        # TODO: mostly the same as get_likelihoods
        posed_mice =  []
        for i,ctx in enumerate(self.contexts):
            ctx.push()
            self.jointRotations_gpu[i].set_async(joint_angles)

            #fk
            self.fk[i](self.baseJointRotations_gpu[i],
                    self.jointRotations_gpu[i],
                    self.jointTranslations_gpu[i],
                    self.jointTransforms_gpu[i],
                    grid=(self.numBlocksFK,1,1),
                    block=(self.numThreadsFK,1,1))

            #skin
            # PLEASE ADD THE ABILITY TO ADD SCALING
            self.skinning[i](self.jointTransforms_gpu[i],
                    self.mouseVertices_gpu[i],
                    self.jointWeights_gpu[i],
                    self.jointWeightIndices_gpu[i],
                    self.skinnedVertices_gpu[i],
                    grid=(self.numBlocksSK,1,1),
                    block=(self.numThreadsSK,1,1))

            #raster
            self.raster[i]( self.skinnedVertices_gpu[i], 
                    self.mouseVertices_gpu[i],
                    self.mouseVertexIdx_gpu[i],
                    self.synthPixels_gpu[i],
                    grid=(self.numBlocksRS,1,1),
                    block=(self.numThreadsRS,1,1))

            posed_mice.append(self.synthPixels_gpu[i].get())
            ctx.pop()
            return np.vstack(posed_mice).reshape(-1,64,64)
        

    def teardown(self):
        # Free everything up after the fact
        for i in range(len(self.contexts)):
            del self.synthPixels_gpu[0]
            del self.realPixels_gpu[0]
            del self.mouseVertices_gpu[0]
            del self.mouseVertexIdx_gpu[0]
            del self.skinnedVertices_gpu[0]
            del self.jointWeights_gpu[0]
            del self.jointWeightIndices_gpu[0]
            del self.jointTransforms_gpu[0]
            del self.likelihoods_gpu[0]
            del self.jointRotations_gpu[0]
            del self.jointTranslations_gpu[0]

        for c in self.contexts:
            driver.Context.pop() # don't know why this is needed, but it seems to be


if __name__ == "__main__":
    from MouseData import MouseData
    import matplotlib.pyplot as plt
    m = MouseData(scenefile="mouse_mesh_low_poly3.npz")
    mp = MousePoser(mouseModel=m, maxNumBlocks=30)
    ja = np.tile(mp.jointRotations_cpu, (1,1))
    ja[:,0] += np.random.normal(size=(ja.shape[0],), scale=10)
    ja[:,2] += np.random.normal(size=(ja.shape[0],), scale=10)
    p = mp.get_posed_mice(ja)
    l = mp.get_likelihoods(ja, np.zeros((64,64), dtype='float32'))
    # plt.imshow(p[:64,:])