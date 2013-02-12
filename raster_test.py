import numpy as np
import time
from pycuda import driver, compiler, gpuarray, tools
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda.driver import func_cache
from MouseData import MouseData

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

# Grab a mouse and its vertices
m = MouseData(scenefile="mouse_mesh_low_poly3.npz")
# Put the vertex data on the GPU
vert_gpu = gpuarray.to_gpu(m.vertices[:,:3].astype('float32'))
vert_idx_gpu = gpuarray.to_gpu(m.vertex_idx.astype('uint16'))

# Cache rules everything around me
preferL1 = True
if preferL1:
    pycuda.autoinit.context.set_cache_config(func_cache.PREFER_L1)
else:
    pycuda.autoinit.context.set_cache_config(func_cache.PREFER_SHARED)

# Go ahead and grab the kernel code
with open("raster_test.cu") as kernel_file:
    kernel_code_template = kernel_file.read()

# In this kernel, currently no formatting
kernel_code = kernel_code_template.format()

# compile the kernel code 
mod = compiler.SourceModule(kernel_code, options=\
                        ['-use_fast_math', \
                        '-I/home/dattalab/Code/cuda-tests/include', \
                        '-I/home/dattalab/Code/cuda-tests/include/Eigen'\
                        '--compiler-options', '-w'], no_extern_c=True)

raster = mod.get_function("RasterKernel")
raster(vert_gpu, vert_idx_gpu, 
        grid=(1,1,1),
        block=(1,1,1)
    )
driver.Context.synchronize()