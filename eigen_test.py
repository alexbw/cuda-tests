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
with open("eigen_test.cu") as kernel_file:
    kernel_code_template = kernel_file.read()

# In this kernel, currently no formatting
kernel_code = kernel_code_template.format()

# compile the kernel code 
mod = compiler.SourceModule(kernel_code, options=\
                        ['-use_fast_math', \
                        '-I/home/dattalab/Code/cuda-tests/include', \
                        '-I/home/dattalab/Code/cuda-tests/include/Eigen',\
                        '--compiler-options', '-w'], no_extern_c=True)

# get the kernel function from the compiled module
eigen = mod.get_function("EigenKernel")
nvmat = mod.get_function("nvMatrixKernel")
rotation = mod.get_function("EigenTransformTest")
idtest = mod.get_function("EigenIdentityTest")
cholesky = mod.get_function("EigenCholeskyTest")

rotation(grid=(1,1,1),block=(1,1,1))
cholesky(grid=(1,1,1),block=(1,1,1))
driver.Context.synchronize()
eigen(grid = (100,1,1), block = (32,1,1))

# idtest(grid=(1,1,1),block=(1,1,1))

# Call the kernel on the card

nvmat_start = time.time()
nvmat(
    grid = (100,1,1),
    block = (32,1,1),
    )
driver.Context.synchronize()
nvmat_time = time.time() - nvmat_start

eigen_start = time.time()
eigen(
    grid = (100,1,1),
    block = (32,1,1),
    )

# Make sure the computation is completed
driver.Context.synchronize()
eigen_time = time.time() - eigen_start




print "Eigen time: {eig}. nvMat time: {nvm}".format(eig=eigen_time, nvm=nvmat_time)