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
with open("matrix_multiply_kernel.cu") as kernel_file:
    kernel_code_template = kernel_file.read()

# In this kernel, currently no formatting
kernel_code = kernel_code_template.format()

# compile the kernel code 
mod = compiler.SourceModule(kernel_code, options=['-use_fast_math'], no_extern_c=True)

# get the kernel function from the compiled module
matrixmul = mod.get_function("MatrixMultiplyKernel")


for num_matrices in [512]:
    for BLOCK_SIZE in [256, 512]:
            out_cpu = np.zeros((BLOCK_SIZE,4,4), dtype='float32')
            out_cpu_test = out_cpu.copy()
            out_gpu = gpuarray.to_gpu(out_cpu)

            start = time.time()
            # Call the kernel on the card
            matrixmul(
                np.int32(num_matrices),
                out_gpu,
                grid = (1,1,1),
                block = (BLOCK_SIZE,1,1),
                )

            # Make sure the computation is completed
            driver.Context.synchronize()
            
            # Pull the data down from the card (if that's part of the benchmark)
            val = out_gpu.get()

            # Hit the stopwatch
            end = time.time()
            gpu_time = end-start

            print "GPU time: {t} ({tsize} threads)".format(tsize=BLOCK_SIZE, t=gpu_time)

            # Do the CPU benchmark
            cpu_start_time = time.time()
            a = np.eye(4)
            for j in range(BLOCK_SIZE):
                for i in range(num_matrices):
                    b = np.eye(4)
                    a = np.dot(a,b)
                out_cpu_test[j,:,:] = a
            cpu_time = time.time() - cpu_start_time
            print "CPU time: {0}".format(cpu_time)
            print "{0}x speed ratio".format(cpu_time/gpu_time)
            print ""

