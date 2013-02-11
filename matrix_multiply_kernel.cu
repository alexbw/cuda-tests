#include <stdio.h>
#include <helper_math.h>
#include <nvMatrix.h>

#define PI 3.14159265359

typedef matrix4<float> matrix4f;

__device__ float deg2rad(float deg) {{
    return deg*PI/180.0;
}}

__device__ float rad2deg(float rad) {{
    return 180.0*rad/PI;
}}

__device__ matrix4f calcLocalRotation(float rotx, float roty, float rotz,
                                      float transx, float transy, float transz) {{


    rotx = deg2rad(rotx);
    roty = deg2rad(roty);
    rotz = deg2rad(rotz);


    float cx = cos(rotx);
    float sx = sin(rotx);
    float cy = cos(roty);
    float sy = sin(roty);
    float cz = cos(rotz);
    float sz = sin(rotz);

    matrix4f Rx; matrix4f Ry; matrix4f Rz;
    Rx.make_identity();
    Ry.make_identity();
    Rz.make_identity();

    Rx.set_scale(cx);
    Ry.set_scale(cy);
    Rz.set_scale(cz);

    Rx(0,0) += 1.0 - cx;
    Ry(1,1) += 1.0 - cy;
    Rz(2,2) += 1.0 - cz;

    Rx(1,2) += -sx;
    Rx(2,1) += sx;

    Ry(0,2) += sy;
    Ry(2,0) += -sy;

    Rz(0,1) += -sz;
    Rz(1,0) += sz;

    matrix4f Tout = Rx*Ry*Rz;

    Tout(0,3) = transx;
    Tout(1,3) = transy;
    Tout(2,3) = transz;

    return Tout;
}}


extern "C"
__global__ void MatrixMultiplyKernel(int numMatrices, float *out)
{{
	// Block index
    const uint bx = blockIdx.x;
    const uint bw = blockDim.x;

    // Thread index
    const uint tx = threadIdx.x;

    // Stride access locations
    const uint aBegin = bw*bx;
    const uint idx = (aBegin+tx)*16; // 4x4 matrix being stored

    matrix4f* J = (matrix4f *)malloc(sizeof(matrix4f)*);
    for (int i=0;i<numMatrices;++i) {{
        J[0] = matrix4f();
    }}

    // // Kick off the joint chain
    // J[0] = calcLocalRotation(0.,0.,0.,1.,1.,1.);

    // for (int i=1;i<numMatrices;++i) {{
    //     // Make a rotation matrix
    //     J[i] = calcLocalRotation(threadIdx.x,0.,0.,1.,1.,1.);
    //     // Multiply it into the last matrix
    //     J[i] = J[i]*J[i-1];
    // }}

    // // Store it all out
    // for (int i=0;i<16;++i) {{
    //     out[idx+i] = J[0]._array[i];
    // }}

}}