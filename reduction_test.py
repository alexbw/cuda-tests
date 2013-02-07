#!/usr/bin/env python

import numpy as np
import time
from pycuda import driver, compiler, gpuarray, tools
from pycuda.compiler import SourceModule
import pycuda.autoinit




with open("test_kernel.cu") as kernel_file:
    kernel_code_template = kernel_file.read()

kernel_code = kernel_code_template.format(message="Awesome!")

# compile the kernel code 
mod = compiler.SourceModule(kernel_code)

# get the kernel function from the compiled module
matrixmul = mod.get_function("MatrixMulKernel")


# call the kernel on the card

gpu_speeds = []
cpu_speeds = []

for MATRIX_SIZE in [4,8,16]:
    for GRID_SIZE in [128,256,512]:

        a_cpu = np.ones((MATRIX_SIZE*GRID_SIZE, MATRIX_SIZE*GRID_SIZE), dtype='float32')
        c_cpu = np.zeros((GRID_SIZE, GRID_SIZE), dtype='float32')

        a_gpu = gpuarray.to_gpu(a_cpu)
        c_gpu = gpuarray.to_gpu(c_cpu)


        start = time.time()
        matrixmul(
            a_gpu,
            c_gpu,
            grid = (GRID_SIZE, GRID_SIZE),
            block = (MATRIX_SIZE, MATRIX_SIZE, 1),
            )
        c_gpu.get()
        end = time.time()

        print "Execution time {msize}/{gsize}: {t}".format(msize=MATRIX_SIZE, gsize=GRID_SIZE, t=end-start)


        accumulator = np.zeros((GRID_SIZE, GRID_SIZE))
        cpu_time = time.time()
        for i in range(GRID_SIZE):
            rbegin = i*MATRIX_SIZE
            rend = rbegin+MATRIX_SIZE
            for j in range(GRID_SIZE):
                cbegin = j*MATRIX_SIZE
                cend = cbegin+MATRIX_SIZE
                accumulator[i,j] = np.sum(a_cpu[rbegin:rend,cbegin:cend])
        print time.time() - cpu_time



# print "-" * 80
# print a_gpu.get()
# 