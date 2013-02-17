#define EIGEN_NO_MALLOC
#define NDEBUG // VERY VERY IMPORTANT FOR PERFORMANCE!!
#include <iostream>
#include <stdio.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <nvMatrix.h>
#include <Eigen/Geometry>
#include <Eigen/Cholesky>

typedef matrix4<float> matrix4f;
#define NITER 10000
using namespace Eigen;
using namespace std;

extern "C"
__global__ void EigenCholeskyTest()
{{

    Affine3f t(Affine3f::Identity());
    t *= Translation3f(0.0, 0.0, 0.0);
    t *= AngleAxisf(30.0, Vector3f::UnitX());
    t *= AngleAxisf(M_PI / 180.0f * 30.0f, Vector3f::UnitY());
    t *= AngleAxisf(30.0, Vector3f::UnitZ());
    t *= Scaling(3.0f, 3.0f, 3.0f);
    Matrix4f D = t.matrix();

    // Matrix4f D = Matrix4f::Random();
    printf("HEY YOU...\n");
    Matrix4f A = D.transpose() * D;
    Vector4f b = D.transpose() * Vector4f(0.5,0.3,0.1,0.5);
    Vector4f x;
    printf("Well, we get this far...\n");
    LLT<Matrix4f> lltOfA(A);
    lltOfA.solveInPlace(b);
    printf("We're running...\n");
}}

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
__global__ void EigenTransformationKernel()
{{

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

extern "C"
__global__ void EigenTransformTest()
{{
    Affine3f t(Affine3f::Identity());
    t *= Translation3f(0.0, 0.0, 0.0);
    t *= AngleAxisf(30.0, Vector3f::UnitX());
    t *= AngleAxisf(M_PI / 180.0f * 30.0f, Vector3f::UnitY());
    t *= AngleAxisf(30.0, Vector3f::UnitZ());
    t *= Scaling(3.0f, 3.0f, 3.0f);
    // for (int i=0; i < 4; ++i) {{
    //     for (int j=0; j < 4; ++j) {{
    //         printf("%f\t", t.matrix()(i,j));
    //     }}
    //     printf("\n");
    // }}
}}

