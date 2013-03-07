// #define EIGEN_NO_MALLOC
#define NDEBUG // VERY VERY IMPORTANT FOR PERFORMANCE!!

// TODO:
// Triangle culling
// - Within bounds?
// - Will be visible?

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
#define NJOINTS {njoints}
#define RESOLUTION_X {resx}
#define RESOLUTION_Y {resy}
#define NUMPIXELS_PER_MOUSE RESOLUTION_X*RESOLUTION_Y

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

struct JointWeights
{{
    float w[N_JOINT_INFLUENCES];
}};

struct JointWeightIndices
{{
    unsigned short idx[N_JOINT_INFLUENCES];
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


__device__ inline void translate(Matrix4f &transform, float transx, float transy, float transz)
{{
	transform(0,3) += transx;
	transform(1,3) += transy;
	transform(2,3) += transz;
}}


__device__ inline Matrix4f scaleMatrix(float scalex, float scaley, float scalez)
{{
	Matrix4f t = Matrix4f::Identity();
	t(0,0) = scalex;
	t(1,1) = scaley;
	t(2,2) = scalez;
	return t;
}}

__device__ Matrix4f calculateEMatrix(GLVertex angle, GLVertex translation)
{{
    float rotx = angle.x;
    float roty = angle.y;
    float rotz = angle.z;

    rotx = deg2rad(rotx);
    roty = deg2rad(roty);
    rotz = deg2rad(rotz);

    float cx = cos(rotx);
    float sx = sin(rotx);
    float cy = cos(roty);
    float sy = sin(roty);
    float cz = cos(rotz);
    float sz = sin(rotz);

    Matrix4f out = Matrix4f::Identity();
    out <<  cy*cz,          cy*sz,         -sy   , translation.x,
            -cx*sz+sx*sy*cz, cx*cz+sx*sy*sz, sx*cy, translation.y,
            sx*sz+sy*cx*cz, -sx*cz+cx*sy*sz,  cx*cy, translation.z,
            0.0,            0.0,              0.0,    1.0;

    return out;

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

__device__ void copyMat4x4ToEigen(Plain4x4Matrix_f plainMat, Matrix4f &EigenMat)
{{
    for (int i=0; i < 4; ++i) {{
        for (int j=0; j < 4; ++j) {{
            int idx = i*4 +j;
            EigenMat(i,j) = plainMat.matrix[idx];
        }}
    }}
}}

__device__ void copyEigenToMat4x4(Plain4x4Matrix_f plainMat, Matrix4f &EigenMat)
{{
    for (int i=0; i < 4; ++i) {{
        for (int j=0; j < 4; ++j) {{
            int idx = i*4 +j;
            plainMat.matrix[idx] = EigenMat(i,j);
        }}
    }}
}}

__device__ void printEigenMat(Matrix4f someMatrix) 
{{
    for (int j=0; j < 4; ++j) {{
        printf("\t%2.3f, %2.3f, %2.3f, %2.3f,\n", 
        someMatrix(j,0),
        someMatrix(j,1),
        someMatrix(j,2),
        someMatrix(j,3));
    }}
    printf("\n");

}}
// ==============================
// Here's the actual work here!
// ==============================


extern "C"
__global__ void rasterizeSerial(GLVertex *skinnedVertices, 
                            GLTriangleFace *triangles,
                            float *synthPixels) 
{{

    const uint bx = blockIdx.x;
    const uint bw = blockDim.x;
    const uint tx = threadIdx.x;



    // Make sure we're looking at the right data
    skinnedVertices += NVERTS*(bw*bx + tx);


    int depthBufferOffset = NUMPIXELS_PER_MOUSE*(bx*bw + tx);

    // For each triangle, rasterize the crap out of it
    // (for now, don't care about overlaps)
    for (int iface=0; iface < NFACES; ++iface) 
    {{
        unsigned short i0 = triangles[iface].v0;
        unsigned short i1 = triangles[iface].v1;
        unsigned short i2 = triangles[iface].v2;

        Vector3f a(skinnedVertices[i0].x, skinnedVertices[i0].y, skinnedVertices[i0].z);
        Vector3f b(skinnedVertices[i1].x, skinnedVertices[i1].y, skinnedVertices[i1].z);;
        Vector3f c(skinnedVertices[i2].x, skinnedVertices[i2].y, skinnedVertices[i2].z);;
        
        Vector3f ll = getLowerLeftOfTriangle(a,b,c);
        Vector3f ur = getUpperRightOfTriangle(a,b,c);

        for (int i=ll(1); i < ur(1); ++i) {{
            for (int j=ll(0); j < ur(0); ++j) {{
                Vector3f pt(j+0.5,i+0.5,0);
                Vector3f baryCoord = calcBarycentricCoordinate(pt,a,b,c);
                bool inTriangle = isBarycentricCoordinateInBounds(baryCoord);

                if (inTriangle) {{
                    float interpZ = getZAtBarycentricCoordinate(baryCoord,a,b,c);
                    long int idx = i*RESOLUTION_X + j;
                    idx += depthBufferOffset;
                    float oldval = synthPixels[idx];
                    if (oldval <= interpZ) {{
                       atomicExch(&synthPixels[idx], interpZ);
                    }}
                }}
                
            }}
        }}
    }}

}}

extern "C"
// Fastest with 10 blocks, 256 threads
// Also, faster than the cache version. 
__global__ void likelihoodSerial(float *synthPixels, 
                            float *realPixels,
                            float *likelihood)
{{
    const uint bx = blockIdx.x;
    const uint bw = blockDim.x;
    const uint tx = threadIdx.x;

    int mouseIdx = bx*bw + tx;
    int synthPixelOffset = NUMPIXELS_PER_MOUSE*mouseIdx;

    float accumulator = 0.0;
    for (int i=0; i < NUMPIXELS_PER_MOUSE; ++i) {{
        accumulator -= abs(realPixels[i] - synthPixels[i+synthPixelOffset]);

    }}

    atomicExch(&likelihood[mouseIdx], accumulator);


}}

extern "C"
// TODO: ROTATIONS
__global__ void skinningSerial(Plain4x4Matrix_f *jointTransforms,
                                GLVertex *vertices,
                                JointWeights *jointWeights,
                                JointWeightIndices *jointWeightIndices,
                                GLVertex *mouseScales,
                                GLVertex *mouseOffsets,
                                GLVertex *mouseRotations,
                                GLVertex *skinnedVertices)
{{

    const uint bx = blockIdx.x;
    const uint bw = blockDim.x;
    const uint tx = threadIdx.x;

    int mouseIdx = bx*bw + tx;
    jointTransforms += mouseIdx*NJOINTS;
    skinnedVertices += mouseIdx*NVERTS;
    GLVertex scale = mouseScales[mouseIdx];
    GLVertex offset = mouseOffsets[mouseIdx];
    GLVertex rotation = mouseRotations[mouseIdx];

    // Calculate a joint's local rotation matrix
    Vector4f vertex;
    vertex << 0.0, 0.0, 0.0, 0.0;

    // Grab the joint transformations, put them in a usable format
    // (All matrix multiplication is done w/ Eigen)
    Matrix4f theseJoints[NJOINTS];
    for (int i=0; i < NJOINTS; ++i) {{
        theseJoints[i] = Matrix4f(jointTransforms[i].matrix);
    }}

    // Precalculate some transformation matrices
    offset.x += RESOLUTION_X/2;
    offset.y += RESOLUTION_Y/2;
    Matrix4f E = calculateEMatrix(rotation, offset);
    // Matrix4f scale_matrix = scaleMatrix(scale.x*RESOLUTION_X, scale.y*RESOLUTION_Y, scale.z);
    Matrix4f scale_matrix = scaleMatrix(scale.x, scale.y, scale.z);
    Vector4f translate_vector(offset.x, offset.y, offset.z, 0.0);

    for (int i=0; i < NVERTS; ++i) {{
        // Grab the unposed vertex
        Vector4f vertex(vertices[i].x, vertices[i].z, vertices[i].y, 1.0);
        // Make our destination vertex
        Vector4f skinnedVertex(0., 0., 0., 0.);

        for (int ijoint=0; ijoint<N_JOINT_INFLUENCES; ++ijoint) {{
            int index = jointWeightIndices[i].idx[ijoint];
            float weight = jointWeights[i].w[ijoint];
            skinnedVertex += weight*theseJoints[index]*vertex;
        }}
        
        // After we've computed the weighted skin position,
        // then we'll scale and translate it into a proper skin space
        skinnedVertex = scale_matrix*skinnedVertex;
        // skinnedVertex = E*skinnedVertex;
        skinnedVertex = skinnedVertex+translate_vector;

        skinnedVertices[i].x = skinnedVertex(0);
        skinnedVertices[i].y = skinnedVertex(1);
        skinnedVertices[i].z = skinnedVertex(2);
    }}


}}


extern "C"
__global__ void FKSerial(GLVertex *baseRotations,
                         GLVertex *rotations,
                         GLVertex *translations,
                         Plain4x4Matrix_f *jointTransforms)
{{
    // NOTE:
    // - The E inverse could be optimized. 

    // Notation:
    // M - skinning matrix. You can multiply an unposed vector into M and get a posed vector.
    // E - local transformation matrix. Represents a rotation and translation from (0,0)
    // "Fixed" matrix - a matrix computed using defualt, or unposed, rotations
    // "Changed" matrix - a matrix computed using non-default, or posed, rotations


    const uint bx = blockIdx.x;
    const uint bw = blockDim.x;
    const uint tx = threadIdx.x;

    int mouseIdx = bx*bw + tx;

    rotations += mouseIdx*NJOINTS;
    jointTransforms += mouseIdx*NJOINTS;

    Matrix4f fixedE[NJOINTS];
    Matrix4f fixedM[NJOINTS];
    Matrix4f changedE[NJOINTS];
    Matrix4f changedM[NJOINTS];
    Matrix4f M[NJOINTS];
    
    // == Get the fixed E's.
    // ========================================
    for (int i=0; i < NJOINTS; ++i) {{
        fixedE[i] = calculateEMatrix(baseRotations[i], translations[i]);
    }}

    // == Get the fixed M's.
    // ========================================
    fixedM[0] = fixedE[0].inverse();
    for (int i=1; i < NJOINTS; ++i) {{
        fixedM[i] = fixedM[i-1]*fixedE[i].inverse();
    }}


    // == Get the Changed E's.
    // ========================================
    for (int i=0; i < NJOINTS; ++i) {{
        changedE[i] = calculateEMatrix(rotations[i], translations[i]);
    }}

    // == Get the changed M's
    // ========================================
    changedM[0] = changedE[0];
    for (int i=1; i < NJOINTS; ++i) {{
        changedM[i] = changedE[i]*changedM[i-1];
    }}

    // == Create the final M's by multiplying the fixed and changed M's. 
    // ========================================
    for (int i=0; i < NJOINTS; ++i) {{
        M[i] = fixedM[i]*changedM[i];
        for (int ii=0; ii < 4; ++ii) {{
            for (int jj=0; jj < 4; ++jj) {{
                int idx = ii*4 + jj;
                jointTransforms[i].matrix[idx] = M[i](ii,jj);
            }}
        }}
    }}


}}







// CODE GRAVEYARD
/*
extern "C"

extern "C"
__global__ void rasterizeParallel(GLVertex *skinnedVertices, 
                            GLVertex *vertices, // TODO: remove once FK and skinning exist.
                            GLTriangleFace *triangles,
                            float *depthBuffer)
{{

    const uint bx = blockIdx.x;
    const uint bw = blockDim.x;
    const uint tx = threadIdx.x;

    // The block determines which primitive we're on
    // printf("Working on triangle %d\n", bx);
    const int primitiveIdx = bx;

    Matrix3f scale_matrix = scaleMatrix3D(RESOLUTION_X*0.3, RESOLUTION_Y*0.3, 24.0);
    Vector3f translate_vector(RESOLUTION_X/2, RESOLUTION_Y/2, 0.);

    __shared__ Vector3f a;
    __shared__ Vector3f b;
    __shared__ Vector3f c;
    __shared__ Vector3f ll;
    __shared__ Vector3f ur;
    __shared__ int boundingBoxWidth;
    __shared__ int boundingBoxHeight;
    __shared__ int numPixelsInBoundingBox;
    __shared__ float *sharedDepthBuffer;

    if (threadIdx.x == 0) {{

        // Grab the triangle indices
        unsigned short i0 = triangles[primitiveIdx].v0;
        unsigned short i1 = triangles[primitiveIdx].v1;
        unsigned short i2 = triangles[primitiveIdx].v2;
    
        // NOTE THE SWAP MANG
        // MAYA's coordinates are left-handed, which I liketh not. 
        a << vertices[i0].x, vertices[i0].y, vertices[i0].z;
        b << vertices[i1].x, vertices[i1].y, vertices[i1].z;
        c << vertices[i2].x, vertices[i2].y, vertices[i2].z;

        // THIS IS A HACK UNTIL FK AND SKINNING ARE IMPLEMENTED
        a = scale_matrix*a;
        a = a+translate_vector;
        b = scale_matrix*b;
        b = b+translate_vector;
        c = scale_matrix*c;
        c = c+translate_vector;

        // Find the bounding box
        ll = getLowerLeftOfTriangle(a,b,c);
        ur = getUpperRightOfTriangle(a,b,c);

        boundingBoxWidth = (int)ceilf(ur(0) - ll(0));
        boundingBoxHeight = (int)ceilf(ur(1) - ll(1));
        numPixelsInBoundingBox = boundingBoxWidth*boundingBoxHeight;

        // Create a space for shared memory to be written to
        sharedDepthBuffer = (float *)malloc(numPixelsInBoundingBox*sizeof(float));

    }}

    __syncthreads();

    for (int i=0; i < numPixelsInBoundingBox; ++i) {{
        sharedDepthBuffer[i] = (float)(i);    }}

    // All threads write into the sharedDepthBuffer
    // for (int i=ll(1); i < ur(1); ++i) {{
    //     for (int j=ll(0); j < ur(0); ++j) {{
    //         Vector3f pt(j+0.5,i+0.5,0);
    //         Vector3f baryCoord = calcBarycentricCoordinate(pt,a,b,c);
    //         bool inTriangle = isBarycentricCoordinateInBounds(baryCoord);

    //         if (inTriangle) {{
    //             float interpZ = getZAtBarycentricCoordinate(baryCoord,a,b,c);
    //             long int idx = i*RESOLUTION_X + j;
    //             float oldval = depthBuffer[idx];
    //             if (oldval <= interpZ) {{
    //                atomicExch(&depthBuffer[idx], interpZ);
    //             }}
    //         }}
            
    //     }}
    // }}

    // __syncthreads();


    // Write out to the depth buffer
    // TODO: should use atomicCAS for this, I believe.
    int counter = 0;
    if (threadIdx.x == 0) {{
        for (int i=ll(1); i < ur(1); ++i) {{
            for (int j=ll(0); j < ur(0); ++j) {{
                long int idx = i*RESOLUTION_X + j;
                float oldVal = depthBuffer[idx];
                float newVal = sharedDepthBuffer[counter];
                newVal = 999.0;
                if (oldVal <= newVal) {{
                    atomicExch(&depthBuffer[idx], newVal);
                }}
                ++counter;
            }}
        }}
    }}

    // __syncthreads();


}}


*/
