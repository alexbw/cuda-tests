#include <stdio.h>

__global__ void SineKernel(float *A)
{{
	// Block index
    const uint bx = blockIdx.x;
    const uint bw = blockDim.x;

    // Thread index
    const uint tx = threadIdx.x;

    // Stride access locations
    const uint aBegin = bw*bx;
    const uint idx = aBegin+tx;

    // Grab a value from global memory
    float this_val = A[idx];
    float sinval = sin(this_val);
    A[idx] = sinval;

    
    // printf("Sin(%f) = %f\n", this_val, sinval);

}}