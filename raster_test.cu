#define EIGEN_NO_MALLOC
#define NDEBUG // VERY VERY IMPORTANT FOR PERFORMANCE!!
#include <iostream>
#include <stdio.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

struct GLVertex
{{
	float x;
	float y;
	float z;
}};

struct GLTriangle
{{
	unsigned short v1;
	unsigned short v2;
	unsigned short v3;
}};

extern "C"
__global__ void RasterKernel(GLVertex *vertices, GLTriangle *triangles)
{{
	for (int i = 0; i < 100; ++i)
	{{	
		// printf("Vert %d: %f,%f,%f\n", 
			// i, vertices[i].x, vertices[i].y, vertices[i].z);
		printf("Tri %d [%d,%d,%d]\n", \
			i, triangles[i].v1, triangles[i].v2, triangles[i].v3);
	}}
	printf("Roger roger\n");
}}