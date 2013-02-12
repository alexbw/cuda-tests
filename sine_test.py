#!/usr/bin/env python

import numpy as np
import time
from pycuda import driver, compiler, gpuarray, tools
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda.driver import func_cache

preferL1 = True
if preferL1:
    pycuda.autoinit.context.set_cache_config(func_cache.PREFER_L1)
else:
    pycuda.autoinit.context.set_cache_config(func_cache.PREFER_SHARED)


# Go ahead and grab the kernel code
with open("sine_kernel.cu") as kernel_file:
    kernel_code_template = kernel_file.read()

# In this kernel, currently no formatting
kernel_code = kernel_code_template.format()

# compile the kernel code 
mod = compiler.SourceModule(kernel_code, options=['-use_fast_math'])

# get the kernel function from the compiled module
matrixmul = mod.get_function("SineKernel")

gpu_speeds = []
cpu_speeds = []

for MATRIX_SIZE in [32, 64, 128]:
    for GRID_SIZE in [1, 10, 100, 1000, 100000]:

        a_cpu = np.pi*0.5*np.ones((MATRIX_SIZE*GRID_SIZE,), dtype='float32')
        a_gpu = gpuarray.to_gpu(a_cpu)

        start = time.time()
        # Call the kernel on the card
        matrixmul(
            a_gpu,
            grid = (GRID_SIZE,),
            block = (MATRIX_SIZE,1,1),
            )
        driver.Context.synchronize()
        # Pull the data down from the card (that's part of the benchmark)
        # val = a_gpu.get()
        end = time.time()
        gpu_time = end-start
        print "Calculating sin of {0} elements".format(GRID_SIZE*MATRIX_SIZE)
        print "GPU time: {t} ({msize} threads/{gsize} blocks)".\
                format(msize=MATRIX_SIZE, gsize=GRID_SIZE, t=gpu_time)

        cpu_start_time = time.time()
        np.sin(a_cpu)
        cpu_time = time.time() - cpu_start_time
        print "CPU time: {0}".format(cpu_time)
        print "{0}x speed ratio".format(cpu_time/gpu_time)
        print ""

