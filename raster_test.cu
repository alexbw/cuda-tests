// #define EIGEN_NO_MALLOC
#define NDEBUG // VERY VERY IMPORTANT FOR PERFORMANCE!!

#include <stdio.h>
#include <algorithm>
#include <Eigen/Dense>
#include <math.h>
#include <vector>
using namespace Eigen;
using namespace std;

#define NVERTS 484
#define NFACES 940
#define N_JOINT_INFLUENCES 4
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

struct JointInfluences4Joints
{{
    float w0;
    float w1;
    float w2;
    float w3;
}};

struct JointIndices4Joints
{{
    unsigned short i0;
    unsigned short i1;
    unsigned short i2;
    unsigned short i3;
}};

struct Plain4x4Matrix_f
{{
    float matrix[16];
}};

struct Plain4x4Matrix_us
{{
    unsigned short matrix[16];
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

__device__ inline Matrix4f EigenMatFromMemory(float *mat4inMemory) 
{{
    Matrix4f m;
    m <<
    mat4inMemory[0], mat4inMemory[1], mat4inMemory[2], mat4inMemory[3], 
    mat4inMemory[4], mat4inMemory[5], mat4inMemory[6], mat4inMemory[7], 
    mat4inMemory[8], mat4inMemory[9], mat4inMemory[10], mat4inMemory[11], 
    mat4inMemory[12], mat4inMemory[13], mat4inMemory[14], mat4inMemory[15];
    return m;
}}

// __device__ inline vector<FK(GLVertex *rotations, )

// ==============================
// Here's the actual work here!
// ==============================


extern "C"
__global__ void RasterKernel(GLVertex *vertices, 
							 GLTriangleFace *triangles,
							 float *depthBuffer,
                             float *mouseImage,
                             JointInfluences4Joints *jointWeights,
                             JointIndices4Joints *jointIndices,
                             Plain4x4Matrix_f *jointWorldMatrix,
                             Plain4x4Matrix_f *inverseBindingMatrix,
                             int numJoints,
							 float resolutionX,
							 float resolutionY )
{{

	Matrix3f transform = scaleMatrix3D(30.0, 30.0, 30.0);
	Vector3f transvec(40.5, 40.5, 0.);

    __shared__ float thisMouse[6400];
    for (int i=0; i<6400;++i) {{
        thisMouse[i] = mouseImage[i];
    }}

    float synthMouse[6400];
    for (int i=0; i<6400;++i) {{
        synthMouse[i] = 0.0f;
    }}


    // Solve some FK, just a lil bit.
    Matrix4f posingMatrix[10];
    for (int i=0; i < numJoints; ++i) {{
        Matrix4f thisJointWorld = EigenMatFromMemory(jointWorldMatrix[i].matrix);
        Matrix4f thisInverseBinding = EigenMatFromMemory(inverseBindingMatrix[i].matrix);
        posingMatrix[i] = thisJointWorld*thisInverseBinding;
    }}

    for (int ijoint=0; ijoint < numJoints; ++ijoint) {{
        printf("Matrix%d\n[\n", ijoint);
            for (int j=0; j < 4; ++j) {{
                printf("%0.2f,%0.2f,%0.2f,%0.2f,\n", 
                    posingMatrix[ijoint](j,0),
                    posingMatrix[ijoint](j,1),
                    posingMatrix[ijoint](j,2),
                    posingMatrix[ijoint](j,3));
            }}
        printf("]\n");
    }}


    // Pre-transform the vertices
    for (int i=0; i < NVERTS; ++i) {{
        // Grab from memory
        Vector3f v(vertices[i].x, vertices[i].z, vertices[i].y);

        // Transform to screen space
        v = transform*v;
        v = v+transvec;

        Vector4f v4;
        v4(0) = v(0); v4(1) = v(1); v4(2) = v(2); v4(3) = 1.0;

        // Skin
        int indices[N_JOINT_INFLUENCES];
        indices[0] = jointIndices[i].i0;
        indices[1] = jointIndices[i].i1;
        indices[2] = jointIndices[i].i2;
        indices[3] = jointIndices[i].i3;
        float weights[N_JOINT_INFLUENCES];
        weights[0] = jointWeights[i].w0;
        weights[1] = jointWeights[i].w1;
        weights[2] = jointWeights[i].w2;
        weights[3] = jointWeights[i].w3;

        Vector4f skinnedVert;
        skinnedVert(0) = 0.0; skinnedVert(1) = 0.0; 
        skinnedVert(2) = 0.0; skinnedVert(3) = 0.0; 

        for (int j=0; j < N_JOINT_INFLUENCES; ++j) {{
            int idx = indices[j];
            float weight = weights[j];
            Matrix4f thisMat = posingMatrix[idx];
            Vector4f thisVec = weight*thisMat*v4;
            skinnedVert = skinnedVert+thisVec;

        }}

        vertices[i].x = skinnedVert(0);
        vertices[i].y = skinnedVert(1);
        vertices[i].z = skinnedVert(2);
    }}

	// For each triangle, rasterize the crap out of it
	// (for now, don't care about overlaps)
	for (int iface=0; iface < NFACES; ++iface) 
	{{
		unsigned short i0 = triangles[iface].v0;
		unsigned short i1 = triangles[iface].v1;
		unsigned short i2 = triangles[iface].v2;

        // NOTE THE SWAP MANG
        // MAYA's coordinates are left-handed, which I liketh not. 
		Vector3f a(vertices[i0].x, vertices[i0].y, vertices[i0].z);
		Vector3f b(vertices[i1].x, vertices[i1].y, vertices[i1].z);;
		Vector3f c(vertices[i2].x, vertices[i2].y, vertices[i2].z);;
		
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
                    float oldval = synthMouse[idx];
                    float compareval = thisMouse[idx];
                    if (oldval <= interpZ) {{
                        synthMouse[idx] = interpZ;
                       // atomicExch(&synthMouse[idx], interpZ-compareval);
                    }}
				}}
				
			}}
		}}

        for (int i=0; i < resolutionX*resolutionY; ++i) {{
            depthBuffer[i] = synthMouse[i];
            mouseImage[i] = synthMouse[i] - thisMouse[i];
        }}

	}}

}}


















