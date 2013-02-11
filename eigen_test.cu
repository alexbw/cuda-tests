#define EIGEN_NO_MALLOC
#define NDEBUG // VERY VERY IMPORTANT FOR PERFORMANCE!!
#include <stdio.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <nvMatrix.h>

typedef matrix4<float> matrix4f;
#define NITER 10000
using namespace Eigen;
using namespace std;

extern "C"
__global__ void IncludeTestKernel()
{{
    // float data[9] = {{1,2,3,4,5,6,7,8,9}};
    // Map<Matrix3f> a(data);
    // a << data;
    // a << 1,0,0,0,1,0,0,0,1;
    // Matrix3f b;
    // b << 2,2,2,2,2,2,2,2,2;
    // Matrix3f c;
    // c = a*b;

    // cout << "YEAHEHAHHEA" << '\n';
}}

extern "C"
__global__ void nvMatrixKernel()
{{
    // Benchmark is to create 100 matrices and multiply them, all in a chain
    // Matrix4f Q;
    matrix4f A;    
    A.make_identity();
    for (int i=0; i < NITER;++i) {{        
        matrix4f B;
        B.make_identity();
        B = B*A;
        A = B;
    }}
}}


extern "C"
__global__ void EigenKernel()
{{
    // Benchmark is to create 100 matrices and multiply them in a chain
    Matrix<float,4,4,RowMajor> A;
    Matrix<float,4,4,RowMajor> C;
    for (int i=0; i < NITER; ++i) {{
        Matrix<float,4,4,RowMajor> B;
        C.noalias() = B*A;
        A.noalias() = C;
    }}
}}


extern "C"
__global__ void EigenIdentityTest()
{{
    Matrix4f A = Matrix4f::Identity();
    A = (Matrix4f::Identity() - A*3.0f)*3.0f;
    printf("What: %f\n", A(0,0));
}}

__global__ void EigenTransformTest()
{{
    Matrix4f A = Matrix4f::Identity();
}}