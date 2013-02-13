#define EIGEN_NO_MALLOC
#define NDEBUG // VERY VERY IMPORTANT FOR PERFORMANCE!!

#include <stdio.h>
#include <algorithm>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

 
#define NVERTS 484
#define NUMTRIS 940
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


__device__ void rotate(Matrix4f &transform, float rotx, float roty, float rotz) {{
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

    transform = transform*Rx*Ry*Rz;
}}

__device__ void translate(Matrix4f &transform, float transx, float transy, float transz)
{{
	transform(0,3) += transx;
    transform(1,3) += transy;
    transform(2,3) += transz;

}}

__device__ void scale(Matrix4f &transform, float scalex, float scaley, float scalez)
{{
	transform(0,0) *= scalex;
	transform(1,1) *= scaley;
	transform(2,2) *= scalez;
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

	Vector3f v0(0.0, 0.0, 0.0);
	Vector3f v1(0.0, 1.0, 0.0);
	Vector3f v2(1.0, 0.0, 0.0);
	Matrix4f transform = Matrix4f::Identity();
	rotate(transform, -90.0, 0.0, 0.0);

	// Matrix4f transform = calcLocalRotation(-90, 0., 0., 0., 0., 0.);

	for (int i=0; i < NVERTS; ++i) {{
		Vector4f v(vertices[i].x, vertices[i].y, vertices[i].z, 1.0);
		Vector4f vout = transform*v;
		vertices[i].x = vout[0];
		vertices[i].y = vout[1];
		vertices[i].z = vout[2];
		// printf("Before: %f. After: %f\n", v[1], vout[1]);
	}}

	Vector3f a(10., 10., 0.2);
	Vector3f b(20., 10., 0.);
	Vector3f c(10., 20., 1.);
	Vector3f pt(0.5, 0.5, 0.);
	Vector3f ll = getLowerLeftOfTriangle(a,b,c);
	Vector3f ur = getUpperRightOfTriangle(a,b,c);


	for (int i=ll(1); i < ur(1); ++i) {{
		for (int j=ll(0); j < ur(0); ++j) {{
			Vector3f pt(i,j,0);
			Vector3f baryCoord = calcBarycentricCoordinate(pt,a,b,c);
			bool inTriangle = isBarycentricCoordinateInBounds(baryCoord);
			if (inTriangle) {{
				float interpZ = getZAtBarycentricCoordinate(baryCoord,a,b,c);
				int idx = i*resolutionX + j;
				depthBuffer[idx] = interpZ;
			}}
		}}
	}}


}}


















