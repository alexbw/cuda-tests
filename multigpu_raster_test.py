# TODO:
# - Better parallel reduction (can work across mice, even)
# - Change likelihoods from numMicePerPass to some multiple thereof,
#       and increment the write location to do multiple passes per frame
# - Pack vertices together as triangles to coalesce transfers
# - Depth culling. If the highest vertex is below the depth at that point, forget it.

# Rasterizer
# Likelihood calculation
# FK
# Skinning
import os
import numpy as np
import time
from pycuda import driver, compiler, gpuarray, tools
from pycuda.compiler import SourceModule
from pycuda.driver import func_cache
from MouseData import MouseData
from matplotlib.pyplot import *
from itertools import product
import fk as forward_kinematics

shouldWeTryFK = True

import pycuda.autoinit
# Grab a context for each GPU
dev = driver.Device(0)
ctx = dev.make_context()
dev.count()
devices = []
contexts = []
devices.append(dev)
contexts.append(ctx)

# Get the number of GPUs in this computer
numGPUs = driver.Device(0).count()

# Grab a context for each GPU
for i in range(1,numGPUs):
    dev = driver.Device(i)
    ctx = dev.make_context()
    devices.append(dev)
    contexts.append(ctx)

# Set the first context as active

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
for ctx in contexts:
    ctx.push()
    if preferL1:
        ctx.set_cache_config(func_cache.PREFER_L1)
    else:
        ctx.set_cache_config(func_cache.PREFER_SHARED)
    ctx.pop()


# Go ahead and grab the kernel code
with open("raster_test.cu") as kernel_file:
    kernel_code_template = kernel_file.read()

# In this kernel, currently no formatting
kernel_code = kernel_code_template.format(resx=resolutionX, 
                                        resy=resolutionY, 
                                        njoints=numJoints)


# compile the kernel code 
basePath = os.path.split(os.path.realpath(__file__))[0]
includePath = os.path.join(basePath, "include")

# Upload data to the GPUs
raster = []
likelihood = []
fk = []
skinning = []

synthPixels_gpu = []
realPixels_gpu = []
mouseVertices_gpu = []
mouseVertexIdx_gpu = []
skinnedVertices_gpu = []
jointWeights_gpu = []
jointWeightIndices_gpu = []
jointTransforms_gpu = []
inverseBindingMatrix_gpu = []
likelihoods_gpu = []
jointRotations_gpu = []
jointTranslations_gpu = []

for ctx in contexts:
    ctx.push()
    mod = compiler.SourceModule(kernel_code, options=\
                            ['-I%s' % includePath, \
                            '--compiler-options', '-w',
                            '--optimize', '3', \
                            ], no_extern_c=True)
    raster.append(mod.get_function("rasterizeSerial"))
    likelihood.append(mod.get_function("likelihoodSerial"))
    fk.append(mod.get_function("FKSerial"))
    skinning.append(mod.get_function("skinningSerial"))


    # Synthetic pixels
    synthPixels_cpu = np.zeros((resolutionX, resolutionY), dtype='float32')
    synthPixels_cpu = np.tile(synthPixels_cpu, (numMicePerPass,1))
    synthPixels_gpu.append(gpuarray.to_gpu(synthPixels_cpu))

    # Real mouse pixels
    realPixels_cpu = np.zeros((int(resolutionX), int(resolutionY)), dtype='float32')
    realPixels_cpu += 10*np.random.random(realPixels_cpu.shape) # testing only
    realPixels_gpu.append(gpuarray.to_gpu(realPixels_cpu))

    # Mouse vertices
    mouseVertices_cpu = m.vertices[:,:3].astype('float32')
    mouseVertices_gpu.append(gpuarray.to_gpu(mouseVertices_cpu))

    # Triangle face indices
    mouseVertexIdx_cpu = m.vertex_idx.astype('uint16')
    mouseVertexIdx_gpu.append(gpuarray.to_gpu(mouseVertexIdx_cpu))

    # Skinned vertices
    skinnedVertices_cpu = mouseVertices_cpu.copy()
    skinnedVertices_cpu = np.tile(skinnedVertices_cpu, (numMicePerPass,1))
    skinnedVertices_gpu.append(gpuarray.to_gpu(skinnedVertices_cpu))

    # Joint weights
    jointWeights_cpu = m.nonzero_joint_weights.astype('float32')
    jointWeights_gpu.append(gpuarray.to_gpu(jointWeights_cpu))

    # Joint weight indices
    jointWeightIndices_cpu = m.joint_idx.astype('uint16')
    jointWeightIndices_gpu.append(gpuarray.to_gpu(jointWeightIndices_cpu))

    # Joint transforms
    if shouldWeTryFK:
        new_rotations = m.joint_rotations.copy()
        new_rotations[2,0] = 30.0
        jointTransforms_cpu = np.vstack(forward_kinematics.get_Ms(new_rotations)).astype('float32')
        jointTransforms_cpu = np.tile(jointTransforms_cpu, (numMicePerPass,1))
    else:
        jointTransforms_cpu = np.eye(4, dtype='float32') # m.jointWorldMatrices
        jointTransforms_cpu = np.tile(jointTransforms_cpu, (numMicePerPass*numJoints,1))
    jointTransforms_gpu.append(gpuarray.to_gpu(jointTransforms_cpu))

    # Inverse binding matrices
    inverseBindingMatrix_cpu = m.inverseBindingMatrices
    inverseBindingMatrix_gpu.append(gpuarray.to_gpu(inverseBindingMatrix_cpu))

    # Likelihoods
    likelihoods_cpu = np.zeros((numMicePerPass,), dtype='float32')
    likelihoods_gpu.append(gpuarray.to_gpu(likelihoods_cpu))

    # Joint rotations
    jointRotations_cpu = m.joint_rotations.astype('float32')
    jointRotations_cpu = jointRotations_cpu[:,[2,1,0]] # WHY? WHY? WHY? WHY? FUCK YOU THAT'S WHY
    jointRotations_cpu = np.tile(jointRotations_cpu, (numMicePerPass,1))
    jointRotations_gpu.append(gpuarray.to_gpu(jointRotations_cpu))

    # Joint translations (we never propose over these)
    jointTranslations_cpu = m.joint_translations.astype('float32')
    jointTranslations_gpu.append(gpuarray.to_gpu(jointTranslations_cpu))

    # Make sure it's all UP THERE
    ctx.synchronize()
    ctx.pop()

# for (numBlocks,numThreads) in product(range(150,300,10), range(9,13)):
for i in range(1,2):
    
    numBlocksFK,numThreadsFK = 10,256
    numMiceFK = numBlocksFK*numThreadsFK
    numBlocksRS,numThreadsRS = 10,256
    numMiceRS = numBlocksRS*numThreadsRS
    numBlocksSK,numThreadsSK = 10,256
    numMiceSK = numBlocksSK*numThreadsSK
    numBlocksLK,numThreadsLK = 10,256
    numMiceLK = numBlocksLK*numThreadsLK
    numMice = min([numMiceFK, numMiceRS, numMiceSK, numMiceLK])

    start = time.time()
    for i,ctx in enumerate(contexts):
        ctx.push()
        #fk (currently broken, but does the right number of operations)
        fk[i](jointRotations_gpu[i],
                jointTranslations_gpu[i],
                inverseBindingMatrix_gpu[i],
                jointTransforms_gpu[i],
                grid=(numBlocksFK,1,1),
                block=(numThreadsFK,1,1))

        #skin
        # PLEASE ADD THE ABILITY TO ADD SCALING
        skinning[i](jointTransforms_gpu[i],
                mouseVertices_gpu[i],
                jointWeights_gpu[i],
                jointWeightIndices_gpu[i],
                skinnedVertices_gpu[i],
                grid=(numBlocksSK,1,1),
                block=(numThreadsSK,1,1))

        #raster
        raster[i]( skinnedVertices_gpu[i], 
                mouseVertices_gpu[i],
                mouseVertexIdx_gpu[i],
                synthPixels_gpu[i],
                grid=(numBlocksRS,1,1),
                block=(numThreadsRS,1,1))

        #likelihood
        likelihood[i](synthPixels_gpu[i],
                realPixels_gpu[i],
                likelihoods_gpu[i],
                grid=(numBlocksLK,1,1),
                block=(numThreadsLK,1,1))
        ctx.pop()

    for ctx in contexts:
        ctx.push()
        ctx.synchronize()
        ctx.pop()

    full_time = time.time() - start
    print "Skin,Raster,Likelihood {micesec} mice/sec".format(micesec=numGPUs*numMice/full_time)


# Do a little display diagnostics
depthBuffer = synthPixels_gpu[0].get()
offset = 0
depthBuffer = depthBuffer[resolutionY*offset:resolutionY*(offset+1),0:resolutionX]
imshow(depthBuffer)
# close('all')
# figure(figsize=(8,3))
# subplot(1,2,1)
# # depthBuffer[depthBuffer == 0] = np.nan
# imshow(depthBuffer)

# subplot(1,2,2)
# realBuffer = realPixels_gpu.get()
# imshow(realBuffer)


l = likelihoods_gpu[0].get()
# assert np.allclose(l[0], np.sum(np.abs(depthBuffer-realBuffer))), "Likelihood gotta be right"





# Free everything up after the fact
for i in range(len(contexts)):
    del synthPixels_gpu[0]
    del realPixels_gpu[0]
    del mouseVertices_gpu[0]
    del mouseVertexIdx_gpu[0]
    del skinnedVertices_gpu[0]
    del jointWeights_gpu[0]
    del jointWeightIndices_gpu[0]
    del jointTransforms_gpu[0]
    del inverseBindingMatrix_gpu[0]
    del likelihoods_gpu[0]
    del jointRotations_gpu[0]
    del jointTranslations_gpu[0]



for c in contexts:
    driver.Context.pop() # don't know why this is needed, but it seems to be






