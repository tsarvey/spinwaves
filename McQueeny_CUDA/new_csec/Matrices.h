#ifndef MATRICES_H
#define MATRICES_H

#include "Numbers.h"
//2D arrays are stored as 1D arrays and this macro is used to index them.  This
//will be faster than an array of pointers when running on the GPU.
//I index them this way as opposed to i*N+j so that I can treat columns as
//vectors in place.  I need to do this more often that I need to do it with
//rows.  Do not change this.  I rely on this storage strategy in several places.
#define INDEX(A,N,i,j) A[i+j*N]

typedef struct
{
	Real x;
	Real y;
	Real z;
} Position;



//For Testing - print a square matrix
void printMat(Complex *M, int n)
{
	int i, j;
	printf("[\n");
	for(i = 0; i < n; i++)
	{
		for(j = 0; j < n; j++)
		{
			printf("%1.5f + %1.5fi,\t", INDEX(M,n,i,j).real, INDEX(M,n,i,j).imag);
		}
		printf(";\n");
	}
	printf("]\n");
}

#endif /* MATRICES_H */
