#define EIGEN_NO_MALLOC
#define NDEBUG // VERY VERY IMPORTANT FOR PERFORMANCE!!

#include <stdio.h>
#include <algorithm>
#include <Eigen/Dense>
#include <math.h>
using namespace Eigen;
using namespace std;

#define NITER 90
#define NVERTS 484
#define NFACES 940
// ==============================
// Helpers!
// ==============================
struct GLVertex
{{
	float x;
	float y;
	float z;
}};

struct BoundingBox
{{
	GLVertex lowerLeft;
	GLVertex upperRight;
}};

struct GLTriangleFace
{{
	unsigned short v0;
	unsigned short v1;
	unsigned short v2;
}};

struct GLTriangle
{{
	GLVertex a;
	GLVertex b;
	GLVertex c;
}};

__device__ inline float deg2rad(float deg)
{{
	return M_PI / 180.0f * deg;
}}

__device__ inline float rad2deg(float rad)
{{
	return rad * 180.0f / M_PI;
}}

__device__ inline Vector3f getLowerLeftOfTriangle(Vector3f a, Vector3f b, Vector3f c)
{{
	float x,y,z;
	x = fminf(fminf(a(0),b(0)),c(0));
	y = fminf(fminf(a(1),b(1)),c(1));
	z = fminf(fminf(a(2),b(2)),c(2));
	return Vector3f(x,y,z);
}}

__device__ inline Vector3f getUpperRightOfTriangle(Vector3f a, Vector3f b, Vector3f c)
{{
	float x,y,z;
	x = fmaxf(fmaxf(a(0),b(0)),c(0));
	y = fmaxf(fmaxf(a(1),b(1)),c(1));
	z = fmaxf(fmaxf(a(2),b(2)),c(2));
	return Vector3f(x,y,z);
}}

// a,b,c are the vertices of the reference triangle
// This takes 3D vectors, because our points are mostly in 3D. It's for convenience, not correctness.
__device__ inline Vector3f calcBarycentricCoordinate(Vector3f vec, Vector3f a, Vector3f b, Vector3f c)
{{
    

    float den = 1 / ((b(1) - c(1)) * (a(0) - c(0)) + (c(0) - b(0)) * (a(1) - c(1)));

    float x = ((b(1) - c(1)) * (vec(0) - c(0)) + (c(0) - b(0)) * (vec(1) - c(1))) * den;
    float y = ((c(1) - a(1)) * (vec(0) - c(0)) + (a(0) - c(0)) * (vec(1) - c(1))) * den;
    float z = 1.0 - x - y;

    return Vector3f(x,y,z);
}}

__device__ inline bool isBarycentricCoordinateInBounds(Vector3f barycentricCoord)
{{

	   return barycentricCoord(0) >= 0.0 && barycentricCoord(0) <= 1.0 &&
          barycentricCoord(1) >= 0.0 && barycentricCoord(1) <= 1.0 &&
          barycentricCoord(2) >= 0.0 && barycentricCoord(2)<= 1.0;

}}

__device__ inline float getZAtBarycentricCoordinate(Vector3f barycentricCoord, Vector3f a, Vector3f b, Vector3f c)
{{
	return barycentricCoord(0)*a(2) + barycentricCoord(1)*b(2) + barycentricCoord(2)*c(2);
}}

__device__ inline Matrix3f rotateMatrix3D(float rotx, float roty, float rotz) {{

	rotx = deg2rad(rotx);
    roty = deg2rad(roty);
    rotz = deg2rad(rotz);

    float cx = cos(rotx);
    float sx = sin(rotx);
    float cy = cos(roty);
    float sy = sin(roty);
    float cz = cos(rotz);
    float sz = sin(rotz);

    Matrix3f Rx = Matrix3f::Identity(); 
    Matrix3f Ry = Matrix3f::Identity(); 
    Matrix3f Rz = Matrix3f::Identity();
    Rx = Rx*cx;
    Ry = Ry*cy;
    Rz = Rz*cz;

    Rx(0,0) += 1.0 - cx;
    Ry(1,1) += 1.0 - cy;
    Rz(2,2) += 1.0 - cz;

    Rx(1,2) += -sx;
    Rx(2,1) += sx;

    Ry(0,2) += sy;
    Ry(2,0) += -sy;

    Rz(0,1) += -sz;
    Rz(1,0) += sz;

    Matrix3f t;
    t = Rx*Ry*Rz;
    return t;
}}

__device__ inline Matrix4f rotateMatrix2(float rotx, float roty, float rotz) {{

	rotx = deg2rad(rotx);
    roty = deg2rad(roty);
    rotz = deg2rad(rotz);

    float cx = cos(rotx);
    float sx = sin(rotx);
    float cy = cos(roty);
    float sy = sin(roty);
    float cz = cos(rotz);
    float sz = sin(rotz);

    Matrix4f Rx = Matrix4f::Identity(); 
    Matrix4f Ry = Matrix4f::Identity(); 
    Matrix4f Rz = Matrix4f::Identity();
    Rx = Rx*cx;
    Ry = Ry*cy;
    Rz = Rz*cz;

    Rx(0,0) += 1.0 - cx;
    Ry(1,1) += 1.0 - cy;
    Rz(2,2) += 1.0 - cz;

    Rx(1,2) += -sx;
    Rx(2,1) += sx;

    Ry(0,2) += sy;
    Ry(2,0) += -sy;

    Rz(0,1) += -sz;
    Rz(1,0) += sz;

    Matrix4f t = Rx*Ry;
    return t;
}}

__device__ inline Matrix4f rotateMatrix(float rotx, float roty, float rotz) {{

    rotx = deg2rad(rotx);
    roty = deg2rad(roty);
    rotz = deg2rad(rotz);

    float cx = cos(rotx);
    float sx = sin(rotx);
    float cy = cos(roty);
    float sy = sin(roty);
    float cz = cos(rotz);
    float sz = sin(rotz);

    Matrix4f Rx = Matrix4f::Identity(); 
    Matrix4f Ry = Matrix4f::Identity(); 
    Matrix4f Rz = Matrix4f::Identity();

    // Right-handed convention
    Rx(1,1) = cx;
    Rx(1,2) = sx;
    Rx(2,1) = -sx;
    Rx(2,2) = cx;

    Ry(0,0) = cy;
    Ry(0,2) = -sy;
    Ry(2,0) = sy;
    Ry(2,2) = cy;

    Rz(0,0) = cz;
    Rz(0,1) = sz;
    Rz(1,0) = -sz;
    Rz(2,2) = cz;

    Matrix4f t = Matrix4f::Identity();
    t = t*Rz*Ry*Rx;
    return t;
}}

__device__ inline void translate(Matrix4f &transform, float transx, float transy, float transz)
{{
	transform(0,3) += transx;
	transform(1,3) += transy;
	transform(2,3) += transz;
}}

__device__ inline Matrix4f translateMatrix(float transx, float transy, float transz)
{{
	Matrix4f t = Matrix4f::Identity();
	t(0,3) = transx;
    t(1,3) = transy;
    t(2,3) = transz;
    return t;

}}

__device__ inline Matrix3f scaleMatrix3D(float scalex, float scaley, float scalez)
{{
	Matrix3f t = Matrix3f::Identity();
	t(0,0) = scalex;
	t(1,1) = scaley;
	t(2,2) = scalez;
	return t;
}}


__device__ inline Matrix4f scaleMatrix(float scalex, float scaley, float scalez)
{{
	Matrix4f t = Matrix4f::Identity();
	t(0,0) = scalex;
	t(1,1) = scaley;
	t(2,2) = scalez;
	return t;
}}

__device__ Matrix4f calcLocalRotation(float rotx, float roty, float rotz,
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

    Matrix4f Rx = Matrix4f::Identity(); 
    Matrix4f Ry = Matrix4f::Identity(); 
    Matrix4f Rz = Matrix4f::Identity();
    Rx = Rx*cx;
    Ry = Ry*cy;
    Rz = Rz*cz;

    Rx(0,0) += 1.0 - cx;
    Ry(1,1) += 1.0 - cy;
    Rz(2,2) += 1.0 - cz;

    Rx(1,2) += -sx;
    Rx(2,1) += sx;

    Ry(0,2) += sy;
    Ry(2,0) += -sy;

    Rz(0,1) += -sz;
    Rz(1,0) += sz;

    Matrix4f Tout = Rx*Ry*Rz;

    Tout(0,3) = transx;
    Tout(1,3) = transy;
    Tout(2,3) = transz;

    return Tout;
}}



// ==============================
// Here's the actual work here!
// ==============================


extern "C"
__global__ void RasterKernel(GLVertex *vertices, 
							 GLTriangleFace *triangles,
							 float *depthBuffer,
							 float resolutionX,
							 float resolutionY )
{{
    

    // Speed challenge
    /*
    Matrix4f mat1 = Matrix4f::Identity();
    mat1 << 0.8413,  0.0004,  0.471 ,  0.757 ,  0.85  ,  0.4208,  0.72  ,
        0.0429,  0.1005,  0.8779,  0.9141,  0.3959,  0.7905,  0.7925,
        0.8939,  0.6818;
    Matrix4f mat2 = Matrix4f::Identity();
    mat2 << 0.8413,  0.0004,  0.471 ,  0.757 ,  0.85  ,  0.4208,  0.72  ,
        0.0429,  0.1005,  0.8779,  0.9141,  0.3959,  0.7905,  0.7925,
        0.8939,  0.6818;
    Matrix4f mat3 = Matrix4f::Identity();
    mat3 << 0.8413,  0.0004,  0.471 ,  0.757 ,  0.85  ,  0.4208,  0.72  ,
        0.0429,  0.1005,  0.8779,  0.9141,  0.3959,  0.7905,  0.7925,
        0.8939,  0.6818;
    #pragma unroll
    for (int i=0; i<NITER;++i) {{
        mat1 = mat2.inverse();
        mat2 = mat1*mat3;
    }}
    */

    Matrix3f swap;
    swap << 1,0,0, \
            0,0,1, \
            0,1,0;

	Matrix3f transform = scaleMatrix3D(30.0, 30.0, 30.0);
	Vector3f transvec(40.5, 40.5, 0.);

	// For each triangle, rasterize the crap out of it
	// (for now, don't care about overlaps)
	for (int iface=0; iface < NFACES; ++iface) 
	{{
		unsigned short i0 = triangles[iface].v0;
		unsigned short i1 = triangles[iface].v1;
		unsigned short i2 = triangles[iface].v2;

		Vector3f a(vertices[i0].x, vertices[i0].z, vertices[i0].y);
		Vector3f b(vertices[i1].x, vertices[i1].z, vertices[i1].y);;
		Vector3f c(vertices[i2].x, vertices[i2].z, vertices[i2].y);;
		
		a = transform*a;
		a = a+transvec;
        b = transform*b;
		b = b+transvec;
        c = transform*c;
        c = c+transvec;

		Vector3f ll = getLowerLeftOfTriangle(a,b,c);
		Vector3f ur = getUpperRightOfTriangle(a,b,c);

		for (int i=ll(1); i < ur(1); ++i) {{
			for (int j=ll(0); j < ur(0); ++j) {{
				Vector3f pt(j+0.5,i+0.5,0);
				Vector3f baryCoord = calcBarycentricCoordinate(pt,a,b,c);
				bool inTriangle = isBarycentricCoordinateInBounds(baryCoord);

				if (inTriangle) {{
					float interpZ = getZAtBarycentricCoordinate(baryCoord,a,b,c);
					long int idx = i*resolutionX + j;
                    float oldval = depthBuffer[idx];
                    if (oldval <= interpZ)
                       atomicExch(&depthBuffer[idx], interpZ);
					   // depthBuffer[idx] = interpZ;
				}}
				
			}}
		}}
	}}

    Matrix4f tr = rotateMatrix(360, 360, 360);
    for (int i=0; i < NVERTS; ++i) {{
        GLVertex v = vertices[i];
        Vector3f a(v.x,v.z,v.y);
        a = transform*a;
        vertices[i].x = a(0);
        vertices[i].y = a(1);
        vertices[i].z = a(2);

    }}



}}


















