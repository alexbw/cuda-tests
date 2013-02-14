import numpy as np
import time
from pycuda import driver, compiler, gpuarray, tools
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda.driver import func_cache
from MouseData import MouseData
from matplotlib.pyplot import *

# Read about this on the "Internets", haven't "tried it" yet.
import cProfile
def profileit(func):
    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profile" # Name the data file sensibly
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(datafn)
        return retval
    return wrapper

# Cache rules everything around me
preferL1 = True
if preferL1:
    pycuda.autoinit.context.set_cache_config(func_cache.PREFER_NONE)
else:
    pycuda.autoinit.context.set_cache_config(func_cache.PREFER_SHARED)

# Go ahead and grab the kernel code
with open("raster_test.cu") as kernel_file:
    kernel_code_template = kernel_file.read()

# In this kernel, currently no formatting
kernel_code = kernel_code_template.format()

# compile the kernel code 
mod = compiler.SourceModule(kernel_code, options=\
                        ['-I/home/dattalab/Code/cuda-tests/include', \
                        '--compiler-options', '-w',
                        '--optimize', '3', \
                        ], no_extern_c=True)

raster = mod.get_function("RasterKernel")

# Grab a mouse and its vertices
m = MouseData(scenefile="mouse_mesh_low_poly3.npz")

# Put the vertex data on the GPU
resolutionX = np.float32(80.0)
resolutionY = np.float32(80.0)
numJoints = np.int32(m.num_joints)
depthBuffer_cpu = np.zeros((resolutionX, resolutionY), dtype='float32')
depthBuffer_gpu = gpuarray.to_gpu(depthBuffer_cpu)
vert_gpu = gpuarray.to_gpu(m.vertices[:,:3].astype('float32'))
vert_idx_gpu = gpuarray.to_gpu(m.vertex_idx.astype('uint16'))
mouse_img_cpu = 10*np.random.random((int(resolutionY), int(resolutionX))).astype("float32")
mouse_img_gpu = gpuarray.to_gpu(mouse_img_cpu)
joint_weights_gpu = gpuarray.to_gpu(m.nonzero_joint_weights.astype('float32'))
joint_indices_gpu = gpuarray.to_gpu(m.joint_idx.astype('uint16'))
joint_world = m.jointWorldMatrices
joint_world_gpu = gpuarray.to_gpu(joint_world)
inverse_binding = m.inverseBindingMatrices
inverse_binding_gpu = gpuarray.to_gpu(inverse_binding)

# Make sure it's all UP THERE
driver.Context.synchronize()

# For-loops for autotuning performance
for num_blocks in [1]:
    for num_threads in [1]:
        for num_repeats in [1]:
            num_mice = num_blocks*num_threads*num_repeats
            raster_start = time.time()    
            for i in range(num_repeats):

                # ACTUALLY CALL THE KERNEL
                raster(vert_gpu, vert_idx_gpu, 
                    depthBuffer_gpu,
                    mouse_img_gpu,
                    joint_weights_gpu,
                    joint_indices_gpu,
                    joint_world_gpu,
                    inverse_binding_gpu,
                    numJoints,
                    resolutionX, resolutionY,
                    grid=(num_blocks,1,1),
                    block=(num_threads,1,1)
                )
            driver.Context.synchronize()
            raster_time = time.time() - raster_start
            print "Rasterized {0} mice/sec [{1}]".format(num_mice/raster_time, num_blocks)

rot_verts = vert_gpu.get()
depthBuffer = depthBuffer_gpu.get()
close('all')
figure(figsize=(8,3))
subplot(1,2,1)
depthBuffer[depthBuffer == 0] = np.nan
imshow(depthBuffer)

subplot(1,2,2)
imshow(mouse_img_gpu.get())