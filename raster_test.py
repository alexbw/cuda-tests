# TODO:
# - Better parallel reduction (can work across mice, even)
# - Free memory after use
# - Change likelihoods from numMicePerPass to some multiple thereof,
#       and increment the write location to do multiple passes per frame
# - Pack vertices together as triangles to coalesce transfers
# - Depth culling. If the highest vertex is below the depth at that point, forget it.

# Rasterizer
# Likelihood calculation
# FK
# Skinning

import numpy as np
import time
from pycuda import driver, compiler, gpuarray, tools
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda.driver import func_cache
from MouseData import MouseData
from matplotlib.pyplot import *
from itertools import product
# First, grab the mouse, and all its wonderful parameters
# Grab a mouse and its vertices
m = MouseData(scenefile="mouse_mesh_low_poly3.npz")

# SET TUNABLE PARAMETERS
numBlocks = 100
numThreads = 512
numMicePerPass = numBlocks*numThreads
resolutionX = np.int32(64)
resolutionY = np.int32(64)
numJoints = m.num_joints


# Cache rules everything around me
preferL1 = False
if preferL1:
    pycuda.autoinit.context.set_cache_config(func_cache.PREFER_L1)
else:
    pycuda.autoinit.context.set_cache_config(func_cache.PREFER_SHARED)


# Go ahead and grab the kernel code
with open("raster_test.cu") as kernel_file:
    kernel_code_template = kernel_file.read()


# In this kernel, currently no formatting
kernel_code = kernel_code_template.format(resx=resolutionX, resy=resolutionY)


# compile the kernel code 
mod = compiler.SourceModule(kernel_code, options=\
                        ['-I/home/dattalab/Code/cuda-tests/include', \
                        '--compiler-options', '-w',
                        '--optimize', '3', \
                        ], no_extern_c=True)
raster = mod.get_function("rasterizeSerial")
likelihood = mod.get_function("likelihoodSerial")


# We need to upload stuff to the graphics card
#
# - Joint rotations (numJoints*3*numMice)
# - Joint translations (numJoints*3*numMice)
# - Mouse vertices (numVerts*3*numMice)
# - Skinned vertices (numVerts*3*numMice)
# - Synth pixels (resX*resY*numMice)
# - Real pixels (resX*resY*numMice)
# - Likelihood (numMice)
# - Joint transforms (4*4*numJoints*numMice)
# - Inverse binding matrices (4*4*numJoints)
#
# But, this is a one-time storage fee, and mice are calculated serially.
# The amount of space reserved can be autotuned.
# Each frame, we only require a host-to-device transfer of 
# - Joint rotations
# - Joint translations
# - Real pixels
# For 5 joints, that's a transfer of
# = 25720 bytes
# = 38 mice transferred/megabyte


# Synthetic pixels
synthPixels_cpu = np.zeros((resolutionX, resolutionY), dtype='float32')
synthPixels_cpu = np.tile(synthPixels_cpu, (numMicePerPass,1))
synthPixels_gpu = gpuarray.to_gpu(synthPixels_cpu)

# Real mouse pixels
realPixels_cpu = np.zeros((int(resolutionX), int(resolutionY)), dtype='float32')
realPixels_cpu += 10*np.random.random(realPixels_cpu.shape) # testing only
realPixels_gpu = gpuarray.to_gpu(realPixels_cpu)

# Mouse vertices
mouseVertices_cpu = m.vertices[:,:3].astype('float32')
mouseVertices_gpu = gpuarray.to_gpu(mouseVertices_cpu)

# Triangle face indices
mouseVertexIdx_cpu = m.vertex_idx.astype('uint16')
mouseVertexIdx_gpu = gpuarray.to_gpu(mouseVertexIdx_cpu)

# Skinned vertices
skinnedVertices_cpu = mouseVertices_cpu.copy()
skinnedVertices_cpu = np.tile(skinnedVertices_cpu, (numMicePerPass,1))
skinnedVertices_gpu = gpuarray.to_gpu(skinnedVertices_cpu)

# Joint weights
jointWeights_cpu = m.nonzero_joint_weights.astype('float32')
jointWeights_gpu = gpuarray.to_gpu(jointWeights_cpu)

# Joint weight indices
jointWeightIndices_cpu = m.joint_idx.astype('uint16')
jointWeightIndices_gpu = gpuarray.to_gpu(jointWeightIndices_cpu)

# Joint transforms
jointTransforms_cpu = np.eye(4, dtype='float32') # m.jointWorldMatrices
jointTransforms_cpu = np.tile(jointTransforms_cpu, (numMicePerPass*numJoints,1))
jointTransforms_gpu = gpuarray.to_gpu(jointTransforms_cpu)

# Inverse binding matrices
inverseBindingMatrix_cpu = m.inverseBindingMatrices
inverseBindingMatrix_gpu = gpuarray.to_gpu(inverseBindingMatrix_cpu)

# Likelihoods
likelihoods_cpu = np.zeros((numMicePerPass,), dtype='float32')
likelihoods_gpu = gpuarray.to_gpu(likelihoods_cpu)

# Joint rotations
jointRotations_cpu = np.zeros((numJoints,3),dtype='float32')
jointRotations_cpu = np.tile(jointRotations_cpu, (numMicePerPass,1))
jointRotations_gpu = gpuarray.to_gpu(jointRotations_cpu)

# Joint translations (we never propose over these)
jointTranslations_cpu = np.zeros((numJoints,3),dtype='float32')
jointTranslations_gpu = gpuarray.to_gpu(jointTranslations_cpu)

# Make sure it's all UP THERE
driver.Context.synchronize()


speeds = []
# [10/128][300/16] is good
# [10/256][300/16]
for (numBlocksRaster, numThreadsRaster, numBlocksLikelihood, numThreadsLikelihood) in product( 
                                    [10], [128, 256],
                                    [250, 300, 350], [8,16]):

    numTimesRedo = 20
    numMicePerPassRaster = numTimesRedo*numBlocksRaster*numThreadsRaster
    numMicePerPassLikelihood = numTimesRedo*numBlocksLikelihood*numThreadsLikelihood
    numMicePerPass = max(numMicePerPassRaster,numMicePerPassLikelihood)
    numLikelihoodPasses = 1
    numRasterPasses = max(1, numMicePerPassLikelihood/numMicePerPassRaster)
    numLikelihoodPasses = max(1, numMicePerPassRaster/numMicePerPassLikelihood)
    print numMicePerPass

    # For-loops for autotuning performance
    raster_start = time.time()    
    for j in range(numTimesRedo):
        # Run the kernel
        for i in range(numRasterPasses):
            raster( skinnedVertices_gpu, 
                    mouseVertices_gpu,
                    mouseVertexIdx_gpu,
                    synthPixels_gpu,
                    grid=(numBlocksRaster,1,1),
                    block=(numThreadsRaster,1,1) )
        for i in range(numLikelihoodPasses):
            likelihood(synthPixels_gpu,
                    realPixels_gpu,
                    likelihoods_gpu,
                    grid=(numBlocksLikelihood,1,1),
                    block=(numThreadsLikelihood,1,1))


    # Make sure the kernel has completed
    driver.Context.synchronize()

    # Hit the stopwatch
    raster_time = time.time() - raster_start
    print "Rasterized {micesec} mice/sec [{br}/{tr}][{bl}/{tl}]".format(micesec=numMicePerPass/raster_time,
                                                            br = numBlocksRaster,
                                                            tr = numThreadsRaster,
                                                            bl = numBlocksLikelihood,
                                                            tl = numThreadsLikelihood)
    benchmark = {
            "threadsRaster":numThreadsRaster,
            "blocksRaster":numBlocksRaster,
            "threadsLikelihood":numThreadsLikelihood,
            "blocksLikelihood":numBlocksLikelihood,
            "micepersec":numMicePerPass
            }
    speeds.append(benchmark)


# Do a little display diagnostics
depthBuffer = synthPixels_gpu.get()
offset = 0
depthBuffer = depthBuffer[resolutionY*offset:resolutionY*(offset+1),0:resolutionX]
close('all')
figure(figsize=(8,3))
subplot(1,2,1)
# depthBuffer[depthBuffer == 0] = np.nan
imshow(depthBuffer)

subplot(1,2,2)
realBuffer = realPixels_gpu.get()
imshow(realBuffer)


l = likelihoods_gpu.get()
assert np.allclose(l[0], np.sum(np.abs(depthBuffer-realBuffer))), "Likelihood gotta be right"





# Free everything up after the fact
del synthPixels_gpu
del realPixels_gpu
del mouseVertices_gpu
del mouseVertexIdx_gpu
del skinnedVertices_gpu
del jointWeights_gpu
del jointWeightIndices_gpu
del jointTransforms_gpu
del inverseBindingMatrix_gpu
del likelihoods_gpu
del jointRotations_gpu
del jointTranslations_gpu











