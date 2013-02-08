#include <stdio.h>
#include <helper_math.h>
#include <nvMatrix.h>
// #include <nvVector.h>

__device__ inline float return_one() 
{{
    return 1.0f;
}}

extern "C"
__global__ void MatrixMulKernel(float *A, float *C)
{{
	// Block index
    const uint bx = blockIdx.x;
    const uint by = blockIdx.y;
    const uint bw = blockDim.x;
    const uint bh = blockDim.y;
    const uint gw = gridDim.x;
    const uint width = gw*bw;

    // Thread index
    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;

    // Stride access locations
    const uint aBegin = bh*width*by + bw*bx;

    // Make sure we have an accumulator
    __shared__ float accumulator;
    if (tx == 0 & ty == 0) accumulator = 0.0;

    // Grab a value from global memory
    float this_val = A[aBegin + width*ty + tx];
    
    // Add our value to the accumulator
    atomicAdd(&accumulator, this_val);
    float one = return_one();
    atomicAdd(&accumulator, one);
    __syncthreads();

    matrix4<float> am(4.0);
    float qqq = am.element(1,1);
    printf("I think some element is: %f\n", qqq);

    // Store out the accumulated value into our global array
    C[bx + by*gw] = accumulator;

}}