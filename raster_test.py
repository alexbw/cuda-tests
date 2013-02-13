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
depthBuffer_cpu = np.zeros((resolutionX, resolutionY), dtype='float32')
depthBuffer_gpu = gpuarray.to_gpu(depthBuffer_cpu)
vert_gpu = gpuarray.to_gpu(m.vertices[:,:3].astype('float32'))
vert_idx_gpu = gpuarray.to_gpu(m.vertex_idx.astype('uint16'))
driver.Context.synchronize()

for num_blocks in [10]:
    for num_threads in [128]:        
        for num_repeats in range(1,5):
            num_mice = num_blocks*num_threads*num_repeats
            raster_start = time.time()    
            for i in range(num_repeats):
                raster(vert_gpu, vert_idx_gpu, 
                    depthBuffer_gpu,
                    resolutionX, resolutionY,
                    grid=(num_blocks,1,1),
                    block=(num_threads,1,1)
                )
            driver.Context.synchronize()
            raster_time = time.time() - raster_start
            print "Rasterized {0} mice/sec [{1}]".format(num_mice/raster_time, num_blocks)

rot_verts = vert_gpu.get()
depthBuffer = depthBuffer_gpu.get()
# clf()
# imshow(depthBuffer)
# colorbar()