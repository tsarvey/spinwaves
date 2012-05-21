#include <stdio.h>
#include <math.h>
#include "Numbers.h"
#include "Matrices.h"
#include "McQueeny_Alg.h"
#include "eigen_decomp.cu"

//take the dot product of two real vectors
__device__ Real realDotProd(Real *a, Real *b, int n)
{
	int i;
	Real result;
	
	result= 0.0;
	
	for(i = 0; i < n; i++)
	{
		result += a[i]*b[i];
	}
	return result;
}


//Cross Section Functions
__device__ int positions_match(Real *pos1, Real *pos2, int magCellSize[3])
{
    Real p11,p12,p13,p21,p22,p23;
	Real mod_sq;
	
	p11 = pos1[0];
	p12 = pos1[1];
	p13 = pos1[2];

	p21 = pos2[0];
	p22 = pos2[1];
	p23 = pos2[2];
	
	
	//shift both positions back to first magCell - assuming they're positive
    while (p11 > magCellSize[0])
        p11 -= magCellSize[0];
    while (p12 > magCellSize[1])
        p12 -= magCellSize[1];
    while (p13 > magCellSize[2])
        p13 -= magCellSize[2];
    while (p21 > magCellSize[0])
        p21 -= magCellSize[0];
    while (p22 > magCellSize[1])
        p22 -= magCellSize[1];
    while (p23 > magCellSize[2])
        p23 -= magCellSize[2];
        
    //now shift negative ones
    while (p11 < 0)
        p11 += magCellSize[0];
    while (p12 < 0)
        p12 += magCellSize[1];
    while (p13 < 0)
        p13 += magCellSize[2];
    while (p21 < 0)
        p21 += magCellSize[0];
    while (p22 < 0)
        p22 += magCellSize[1];
    while (p23 < 0)
        p23 += magCellSize[2];
        

    p11 = p11-p21;
	p12 = p12-p22;
	p13 = p13-p23;
	
	mod_sq = p11*p11 + p12*p12 + p13*p13;
	//sloppy way of finding equality - the only reason this is kindof okay is
	//becuase I know that all the position numbers are on the order of 1.
    if(mod_sq < 1e-10)//I think this is pretty generous
    {
        return 1;
    }
    return 0;
}

/*
This function contructs the matrix used in Mcqueeny's Algorithm.

This is the equation for the i,jth element:
M_ij = delta_ij*Sum_lk{J(0i,lk)*sigma_k*S_k} - sigma_j*sqrt(S_i*S_j)*Sum_l{J(0i,lk)*exp{iq.l}}

-n is the number of atoms in the magnetic cell (so the resulting matrix will be n*n).

-M is the n*n matrix that will be populated by this function.
*/
__device__ void construct_M(Complex *M, int n, Atom *atoms, int magCellSize[3], Real q[3])
{
	int i,j,k;
	Real J, tmp, tmp2;
	Atom nbr;

	//Populate with zeros first
	for(i = 0; i < n; i++)
	{
		for(j = 0; j < n; j++)
		{
			INDEX(M,n,i,j).real = 0.0;
			INDEX(M,n,i,j).imag = 0.0;
		}
	}

    //I'll do the first summation first - diagonal terms
    for(i = 0; i < n; i++)
	{
        for(j = 0; j < atoms[i].numNeighbors; j++)
        {
			J = atoms[i].interactions[j];
			nbr = atoms[atoms[i].neighbors[j]];
			INDEX(M,n,i,i).real += J * nbr.sigma * nbr.S;
		}
	}

	//Now for the rest of the terms...
	for(i = 0; i < n; i++)
	{
		for(j = 0; j < n; j++)
		{
			tmp = atoms[j].sigma * sqrt(atoms[i].S * atoms[j].S);
			for(k = 0; k < atoms[i].numNeighbors; k++)
			{
				//Now I want to find the jth atom in the lth cell
            	//I'll lop through the interactions with i, and if any coincides
            	//with the jth atom, I'll use that
				nbr = atoms[atoms[i].neighbors[k]];
				
				if(positions_match(atoms[j].pos, nbr.pos, magCellSize))//check if this is atom j in any cell
				{
					J = atoms[i].interactions[k];
					tmp = atoms[j].sigma * sqrt(atoms[i].S * atoms[j].S) * J;
					tmp2 = q[0]*nbr.l[0] + q[1]*nbr.l[1] + q[2]*nbr.l[2];
					INDEX(M,n,i,j).real -= tmp * cos(tmp2);
					INDEX(M,n,i,j).imag -= tmp * sin(tmp2);
				}
			}
		}
	}
}

/*

Calculate the cross section at a single q point.

Right now this just computes the structure factor - no form factors, etc. 

-n is the number of atoms int the magnetic unit cell
-atoms is the array of Atom structures
-magCellSize is an array of length 3 giving the dimensions of the magnetic
unit cell in dimensions of crystallographic unit cells (thats why its an int[])
-mem is a big hunk of memory used for scratch work of size:
(6n^2 + 9n) * sizeof(Real)

This function computes S(Q,w) (w here is the energy your looking at) and returns
it.
*/
__device__ Real calcPoint(Real q[3], int n, Atom *atoms, int magCellSize[3], Real *mem)
{
	int memIndex, i, j;
	Complex *M;//our matrix which we will decompose
	Complex *V;//where the eigenvectors will go
	Complex *eigenValues;//where the eigenvalues will go
	Complex c_tmp1, c_tmp2, T_ni;
	Real r_tmp1, r_tmp2, strfac;
	
	//Eigendecomposition settings
	Real epsilon = EPS_M*2;//Probably not ideal, but I think this should be fine
	int maxiter = 50;
	
	//scratch variables used for the eigendecomposition
	Complex *Q;
	Real *rv1;
	Complex *cv1, *cv2, *cv3;
	
	//break up mem
	M = (Complex *)mem;
	memIndex = 2*n*n;
	V = (Complex *) &mem[memIndex];
	memIndex += 2*n*n;
	Q = (Complex *) &mem[memIndex];
	memIndex += 2*n*n;
	eigenValues = (Complex *) &mem[memIndex];
	memIndex += 2*n;
	rv1 = &mem[memIndex];
	memIndex += n;
	cv1 = (Complex *) &mem[memIndex];
	memIndex += 2*n;
	cv2 = (Complex *) &mem[memIndex];
	memIndex += 2*n;
	cv3 = (Complex *) &mem[memIndex];
	memIndex += 2*n;
	
	
	//Construct the matrix whose eigenvalues/vectors we will be using
	construct_M(M, n, atoms, magCellSize, q);
	
	
	//Get the eigenvalues and vectors
	eigenDecomp(M, V, Q, eigenValues, rv1, cv1, cv2, cv3, n, epsilon, maxiter);
	
	
	//Normalize the eigenvectors according to McQueeny's Paper
	//(copying from my python code)
	for(i = 0; i < n; i++)
	{
		r_tmp1 = ABS(eigenValues[i]);
		r_tmp2 = eigenValues[i].real/r_tmp1;//eigevalues should all be real
		r_tmp1 = 0.0;
		
		
		for(j = 0; j < n; j++)
		{
			c_tmp1 = INDEX(V, n, j, i);
			r_tmp1 += atoms[j].sigma * (c_tmp1.real*c_tmp1.real + c_tmp1.imag*c_tmp1.imag);
		}
		
		r_tmp1 = r_tmp2/r_tmp1;
		r_tmp1 = sqrt(r_tmp1);
		
		for(j = 0; j < n; j++)
		{
			INDEX(Q,n,j,i) = INDEX(V,n,j,i);
			INDEX(Q,n,j,i).real *= r_tmp1;
			INDEX(Q,n,j,i).imag *= r_tmp1;
		}
	}
	
	//Now I need to actually calcule the structure factor
	strfac = 0.0;
	for(i = 0; i < n; i++)
	{
		c_tmp1.real = 0.0;
		c_tmp1.imag = 0.0;
		for(j = 0 ; j < n; j++)
		{
			r_tmp1 = realDotProd(q, atoms[j].d, 3);
			c_tmp2.real = cos(-r_tmp1);
			c_tmp2.imag = sin(-r_tmp1);
			T_ni.real = INDEX(Q,n,j,i).real;
			T_ni.imag = INDEX(Q,n,j,i).imag;//make sure you have indice order correct
			c_tmp2 = complexMult(T_ni, c_tmp2);
			r_tmp2 = atoms[j].sigma * sqrt(atoms[j].S);
			c_tmp2.real *= r_tmp2;
			c_tmp2.imag *= r_tmp2;
			
			c_tmp1.real += c_tmp2.real;
			c_tmp1.imag += c_tmp2.imag;
		}
		
		strfac += c_tmp1.real*c_tmp1.real + c_tmp1.imag*c_tmp1.imag;
	}
	
	return strfac;
}

/*
 * This function calles the calPoint function, passing the correct sections of
 * memory allocated on the card.
 */
__global__ void cSecCaller(Atom *atoms, int n, Real *Qx, Real *Qy, Real *Qz, Real *results, Real *mem, int magCellSizeA, int magCellSizeB, int magCellSizeC)
{
	int magCellSize[3];
	Real q[3];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;//thread id used as an index


	//just for convenience	
	magCellSize[0] = magCellSizeA;
	magCellSize[1] = magCellSizeB;
	magCellSize[2] = magCellSizeC;
	q[0] = Qx[tid];
	q[1] = Qy[tid];
	q[2] = Qz[tid];

	results[tid] = calcPoint(q, n, atoms, magCellSize, &mem[(6*n*n + 9*n)*tid]);
}

/*
 * This function will calculate the cross section for numQPoints points and put
 * the result in results.  It only does it in one batch - allocating the memory,
 * doing the computation, and freeing the memory.  Does not yet deal with sets
 * of data too large to fit on the card at once. 
 *
 * -n is the number of atoms in the first magnetic unit cell, but atomListLen
 * is the length of the entire aomt list, which also contains atoms neighboring
 * the first cell.
 *
 * Right now the number of blocks is numQPoints/100 and hte numbe rof threads
 * per block is 100.  Therefore, numQPoints must be divisible by 100, or there
 * will be residual points.
 */
__host__ void cSection(Atom *atoms, int n, int atomListLen, Real *Qx, Real *Qy, Real *Qz, int numQPoints, Real *results, int magCellSize[3])
{
	int sizeCnt, i;
	Real *dev_results;
	Real *dev_Qx;
	Real *dev_Qy;
	Real *dev_Qz;
	Atom *dev_atoms;
	Real *dev_mem;//scratchwork memory on the GPU
	int *dev_nbr;
	Real *dev_int;
	Atom *atoms_cpy = (Atom*)malloc(atomListLen*sizeof(Atom));

	//Make a copy of the atoms list that I can modify before passing to GPU
	for(i = 0; i < atomListLen; i++)
	{
		atoms_cpy[i] = atoms[i];
	}

	printf("Computing %d Q points.  This requires %d bytes of scratch work space on the GPU.\n", numQPoints, numQPoints * (6*n*n+9*n) * sizeof(Real));

	//Allocate memory on the GPU
	HANDLE_ERROR( cudaMalloc( (void**)&dev_results, numQPoints * sizeof(Real) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_Qx, numQPoints * sizeof(Real) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_Qy, numQPoints * sizeof(Real) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_Qz, numQPoints * sizeof(Real) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_atoms, atomListLen * sizeof(Atom) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_mem, numQPoints * (6*n*n+9*n) * sizeof(Real) ) );

	//Allocate and populate the arrays contained in the atom structure
	for(i = 0; i < atomListLen; i++)
	{
		//allocate and populate the neighbors array
		HANDLE_ERROR( cudaMalloc( (void**)&dev_nbr, atoms[i].numNeighbors * sizeof(int) ) );
		HANDLE_ERROR( cudaMemcpy( dev_nbr, atoms[i].neighbors, atoms[i].numNeighbors * sizeof(int), cudaMemcpyHostToDevice ) );
		atoms_cpy[i].neighbors = dev_nbr;

		//allocate and populate the interactions array
		HANDLE_ERROR( cudaMalloc( (void**)&dev_int, atoms[i].numNeighbors * sizeof(Real) ) );
		HANDLE_ERROR( cudaMemcpy( dev_int, atoms[i].interactions, atoms[i].numNeighbors * sizeof(Real), cudaMemcpyHostToDevice ) );
		atoms_cpy[i].interactions = dev_int;
	}

	sizeCnt = numQPoints * sizeof(Real) + numQPoints * 3 * sizeof(Real) + n * sizeof(Atom) + numQPoints * (6*n*n+9*n) * sizeof(Real);
	printf("Total of %d bytes allocated on the GPU.\n", sizeCnt);

	//Copy atoms and q points over to the GPU
	HANDLE_ERROR( cudaMemcpy( dev_atoms, atoms_cpy, atomListLen * sizeof(Atom), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_Qx, Qx, numQPoints * sizeof(Real) , cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_Qy, Qy, numQPoints * sizeof(Real) , cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_Qz, Qz, numQPoints * sizeof(Real) , cudaMemcpyHostToDevice ) );

	printf("Calculating cross section...\n");

	//Run numQPoints instances of the algorithm on the GPU
	//The division by 100 is arbitrary and only there becuase there is a limit
	//on the number of blocks (I beleive 65000).  Obviously, numQPoints must be
	//divisible for 100 for this to calculate all points.
	cSecCaller<<<numQPoints/100,100>>>(dev_atoms, n, dev_Qx, dev_Qy, dev_Qz, dev_results, dev_mem, magCellSize[0], magCellSize[1], magCellSize[2]);

	//I beleive this will keep the host code from executing beyond this point
	//until the GPU code is done.	
	HANDLE_ERROR( cudaThreadSynchronize() );


	//Copy the result back to the host computer
    HANDLE_ERROR( cudaMemcpy( results, dev_results, numQPoints * sizeof(Real), cudaMemcpyDeviceToHost ) );

	//Free the memory allocated on the device
	HANDLE_ERROR( cudaFree( dev_results ) );
	HANDLE_ERROR( cudaFree( dev_Qx ) );
	HANDLE_ERROR( cudaFree( dev_Qy ) );
	HANDLE_ERROR( cudaFree( dev_Qz ) );

	for(i = 0; i < atomListLen; i++)
	{
		//free the neighbors array
		HANDLE_ERROR( cudaFree( atoms_cpy[i].neighbors ) );

		//free the interactions array
		HANDLE_ERROR( cudaFree( atoms_cpy[i].interactions ) );
	}

	HANDLE_ERROR( cudaFree( dev_atoms ) );
	HANDLE_ERROR( cudaFree( dev_mem ) );

	free(atoms_cpy);
}


//------------------------------  Testing  -------------------------------------


/*  ============ CPU =============

//A simple ferro test to make sure I get the same thing as my python code
//Tested and working 5/11/12 - back when this was only running on the CPU
void test1(Real q[3])
{
	Complex M[1];
	int magCellSize[3] = {1,1,1};

	Atom atoms[3];
	int nbrs1[2] = {1,2};
	int nbrs2[2] = {0};
	int nbrs3[1] = {0};
    Real int1[2] = {1.0,1.0};
	Real int2[2] = {1.0,1.0};
	Real int3[1] = {1.0};
	
	atoms[0].numNeighbors = 2;
	atoms[0].neighbors = nbrs1;
    atoms[0].interactions = int1;
    atoms[0].S = 1.0;
    atoms[0].sigma = 1;
    atoms[0].pos[0] = 0.5;
    atoms[0].pos[1] = 0.5;
    atoms[0].pos[2] = 0.5;
    atoms[0].l[0] = 0;
    atoms[0].l[1] = 0;
    atoms[0].l[2] = 0;

	atoms[1].numNeighbors = 1;
	atoms[1].neighbors = nbrs2;
    atoms[1].interactions = int2;
    atoms[1].S = 1.0;
    atoms[1].sigma = 1;
    atoms[1].pos[0] = 1.5;
    atoms[1].pos[1] = 0.5;
    atoms[1].pos[2] = 0.5;
    atoms[1].l[0] = 1;
    atoms[1].l[1] = 0;
    atoms[1].l[2] = 0;
    

	atoms[2].numNeighbors = 1;
	atoms[2].neighbors = nbrs3;
    atoms[2].interactions = int3;
    atoms[2].S = 1.0;
    atoms[2].sigma = 1;
    atoms[2].pos[0] = -0.5;
    atoms[2].pos[1] = 0.5;
    atoms[2].pos[2] = 0.5;
    atoms[2].l[0] = -1;
    atoms[2].l[1] = 0;
    atoms[2].l[2] = 0;

	construct_M(M, 1, atoms, magCellSize, q);
	printMat(M, 1);
}

//A simple anti-ferro test to make sure I get the same thing as my python code
//Tested and working 5/11/12
void test2(Real q[3])
{
	#define N 2
	Complex M[N*N];
	int magCellSize[3] = {2,1,1};

	Atom atoms[4];
	int nbrs1[2] = {1,2};
	int nbrs2[2] = {0,3};
	int nbrs3[1] = {0};
	int nbrs4[1] = {1};
    Real int1[2] = {-1.0,-1.0};
	Real int2[2] = {-1.0,-1.0};
	Real int3[1] = {-1.0};
	Real int4[1] = {-1.0};
	
	atoms[0].numNeighbors = 2;
	atoms[0].neighbors = nbrs1;
    atoms[0].interactions = int1;
    atoms[0].S = 1.0;
    atoms[0].sigma = 1;
    atoms[0].pos[0] = 0.5;
    atoms[0].pos[1] = 0.5;
    atoms[0].pos[2] = 0.5;
    atoms[0].l[0] = 0;
    atoms[0].l[1] = 0;
    atoms[0].l[2] = 0;

	atoms[1].numNeighbors = 2;
	atoms[1].neighbors = nbrs2;
    atoms[1].interactions = int2;
    atoms[1].S = 1.0;
    atoms[1].sigma = -1;
    atoms[1].pos[0] = 1.5;
    atoms[1].pos[1] = 0.5;
    atoms[1].pos[2] = 0.5;
    atoms[1].l[0] = 0;
    atoms[1].l[1] = 0;
    atoms[1].l[2] = 0;
    

	atoms[2].numNeighbors = 1;
	atoms[2].neighbors = nbrs3;
    atoms[2].interactions = int3;
    atoms[2].S = 1.0;
    atoms[2].sigma = -1;
    atoms[2].pos[0] = -0.5;//Tested and working 5/11/12
    atoms[2].pos[1] = 0.5;
    atoms[2].pos[2] = 0.5;
    atoms[2].l[0] = -1;
    atoms[2].l[1] = 0;
    atoms[2].l[2] = 0;
    
    atoms[3].numNeighbors = 1;
	atoms[3].neighbors = nbrs4;
    atoms[3].interactions = int4;
    atoms[3].S = 1.0;
    atoms[3].sigma = 1;
    atoms[3].pos[0] = 2.5;
    atoms[3].pos[1] = 0.5;
    atoms[3].pos[2] = 0.5;
    atoms[3].l[0] = 1;
    atoms[3].l[1] = 0;
    atoms[3].l[2] = 0;

	construct_M(M, N, atoms, magCellSize, q);
	printMat(M, N);
}

*/


//A simple anti-ferro test to make sure I get the same thing as my python code
//Tested and working 5/11/12 up to normalizing the eigenvectors 
//(just the sign on the eigenvectors is off)
//On 5/18/2012 this was test with the entire strfac calculation and it produced
//The same results as my python code (this is a simple anti-ferromagnet).
//This was tested on the CPU, since then it has been converted to run on the GPU.
//On 5/21/12 this test is working on the GPU.
void test3()
{
	#define N 2
	Complex M[N*N];
	int magCellSize[3] = {2,1,1};
	Real *mem = (Real *) malloc((6*N*N + 9*N)*sizeof(Real));

	//Test GPU
	int i;
	#define NUMPOINTS 200000
	Real Qx[NUMPOINTS];
	Real Qy[NUMPOINTS];
	Real Qz[NUMPOINTS];
	Real results[NUMPOINTS];
	for(i = 0; i < NUMPOINTS; i++)
	{
		Qx[i] = 0.000033*i;
		Qy[i] = 0.0;
		Qz[i] = 0.0;
	}
	printf("successfully allocated Q and result arrays.\n");

	Atom atoms[4];
	int nbrs1[2] = {1,2};
	int nbrs2[2] = {0,3};
	int nbrs3[1] = {0};
	int nbrs4[1] = {1};
    Real int1[2] = {-1.0,-1.0};
	Real int2[2] = {-1.0,-1.0};
	Real int3[1] = {-1.0};
	Real int4[1] = {-1.0};
	
	atoms[0].numNeighbors = 2;
	atoms[0].neighbors = nbrs1;
    atoms[0].interactions = int1;
    atoms[0].S = 1.0;
    atoms[0].sigma = 1;
    atoms[0].pos[0] = 0.5;
    atoms[0].pos[1] = 0.5;
    atoms[0].pos[2] = 0.5;
    atoms[0].l[0] = 0;
    atoms[0].l[1] = 0;
    atoms[0].l[2] = 0;
    atoms[0].d[0] = atoms[0].pos[0]/magCellSize[0];
    atoms[0].d[1] = atoms[0].pos[1]/magCellSize[1];
    atoms[0].d[2] = atoms[0].pos[2]/magCellSize[2];

	atoms[1].numNeighbors = 2;
	atoms[1].neighbors = nbrs2;
    atoms[1].interactions = int2;
    atoms[1].S = 1.0;
    atoms[1].sigma = -1;
    atoms[1].pos[0] = 1.5;
    atoms[1].pos[1] = 0.5;
    atoms[1].pos[2] = 0.5;
    atoms[1].l[0] = 0;
    atoms[1].l[1] = 0;
    atoms[1].l[2] = 0;
    atoms[1].d[0] = atoms[1].pos[0]/magCellSize[0];
    atoms[1].d[1] = atoms[1].pos[1]/magCellSize[1];
    atoms[1].d[2] = atoms[1].pos[2]/magCellSize[2];
    
	atoms[2].numNeighbors = 1;
	atoms[2].neighbors = nbrs3;
    atoms[2].interactions = int3;
    atoms[2].S = 1.0;
    atoms[2].sigma = -1;
    atoms[2].pos[0] = -0.5;//Tested and working 5/11/12
    atoms[2].pos[1] = 0.5;
    atoms[2].pos[2] = 0.5;
    atoms[2].l[0] = -1;
    atoms[2].l[1] = 0;
    atoms[2].l[2] = 0;
    atoms[2].d[0] = atoms[2].pos[0]/magCellSize[0];
    atoms[2].d[1] = atoms[2].pos[1]/magCellSize[1];
    atoms[2].d[2] = atoms[2].pos[2]/magCellSize[2];

    atoms[3].numNeighbors = 1;
	atoms[3].neighbors = nbrs4;
    atoms[3].interactions = int4;
    atoms[3].S = 1.0;
    atoms[3].sigma = 1;
    atoms[3].pos[0] = 2.5;
    atoms[3].pos[1] = 0.5;
    atoms[3].pos[2] = 0.5;
    atoms[3].l[0] = 1;
    atoms[3].l[1] = 0;
    atoms[3].l[2] = 0;
    atoms[3].d[0] = atoms[3].pos[0]/magCellSize[0];
    atoms[3].d[1] = atoms[3].pos[1]/magCellSize[1];
    atoms[3].d[2] = atoms[3].pos[2]/magCellSize[2];

	cSection(atoms, N, 4, Qx, Qy, Qz, NUMPOINTS, results, magCellSize);

	for(i = 0; i < NUMPOINTS; i++)
	{
		printf("q: (%f, %f, %f)   S: %f\n", Qx[i], Qy[i], Qz[i], results[i]);
	}
}

//Timing test
void test4()
{
	#define N 2
	Complex M[N*N];
	int magCellSize[3] = {2,1,1};
	Real *mem = (Real *) malloc((6*N*N + 9*N)*sizeof(Real));

	//linux wall time
	struct timeval tvBegin, tvEnd, tvDiff;
	FILE *file; 

	//Test GPU
	int i, j;
	#define NUMPOINTS 150000
	Real Qx[NUMPOINTS];
	Real Qy[NUMPOINTS];
	Real Qz[NUMPOINTS];
	Real results[NUMPOINTS];
	for(i = 0; i < NUMPOINTS; i++)
	{
		Qx[i] = 0.000033*i;
		Qy[i] = 0.0;
		Qz[i] = 0.0;
	}
	printf("successfully allocated Q and result arrays.\n");

	Atom atoms[4];
	int nbrs1[2] = {1,2};
	int nbrs2[2] = {0,3};
	int nbrs3[1] = {0};
	int nbrs4[1] = {1};
    Real int1[2] = {-1.0,-1.0};
	Real int2[2] = {-1.0,-1.0};
	Real int3[1] = {-1.0};
	Real int4[1] = {-1.0};
	
	atoms[0].numNeighbors = 2;
	atoms[0].neighbors = nbrs1;
    atoms[0].interactions = int1;
    atoms[0].S = 1.0;
    atoms[0].sigma = 1;
    atoms[0].pos[0] = 0.5;
    atoms[0].pos[1] = 0.5;
    atoms[0].pos[2] = 0.5;
    atoms[0].l[0] = 0;
    atoms[0].l[1] = 0;
    atoms[0].l[2] = 0;
    atoms[0].d[0] = atoms[0].pos[0]/magCellSize[0];
    atoms[0].d[1] = atoms[0].pos[1]/magCellSize[1];
    atoms[0].d[2] = atoms[0].pos[2]/magCellSize[2];

	atoms[1].numNeighbors = 2;
	atoms[1].neighbors = nbrs2;
    atoms[1].interactions = int2;
    atoms[1].S = 1.0;
    atoms[1].sigma = -1;
    atoms[1].pos[0] = 1.5;
    atoms[1].pos[1] = 0.5;
    atoms[1].pos[2] = 0.5;
    atoms[1].l[0] = 0;
    atoms[1].l[1] = 0;
    atoms[1].l[2] = 0;
    atoms[1].d[0] = atoms[1].pos[0]/magCellSize[0];
    atoms[1].d[1] = atoms[1].pos[1]/magCellSize[1];
    atoms[1].d[2] = atoms[1].pos[2]/magCellSize[2];
    
	atoms[2].numNeighbors = 1;
	atoms[2].neighbors = nbrs3;
    atoms[2].interactions = int3;
    atoms[2].S = 1.0;
    atoms[2].sigma = -1;
    atoms[2].pos[0] = -0.5;//Tested and working 5/11/12
    atoms[2].pos[1] = 0.5;
    atoms[2].pos[2] = 0.5;
    atoms[2].l[0] = -1;
    atoms[2].l[1] = 0;
    atoms[2].l[2] = 0;
    atoms[2].d[0] = atoms[2].pos[0]/magCellSize[0];
    atoms[2].d[1] = atoms[2].pos[1]/magCellSize[1];
    atoms[2].d[2] = atoms[2].pos[2]/magCellSize[2];

    atoms[3].numNeighbors = 1;
	atoms[3].neighbors = nbrs4;
    atoms[3].interactions = int4;
    atoms[3].S = 1.0;
    atoms[3].sigma = 1;
    atoms[3].pos[0] = 2.5;
    atoms[3].pos[1] = 0.5;
    atoms[3].pos[2] = 0.5;
    atoms[3].l[0] = 1;
    atoms[3].l[1] = 0;
    atoms[3].l[2] = 0;
    atoms[3].d[0] = atoms[3].pos[0]/magCellSize[0];
    atoms[3].d[1] = atoms[3].pos[1]/magCellSize[1];
    atoms[3].d[2] = atoms[3].pos[2]/magCellSize[2];

//	cSection(atoms, N, 4, Qx, Qy, Qz, NUMPOINTS, results, magCellSize);

	//Timing test
	file = fopen("times2.txt","w");
	for(i = 1000; i < NUMPOINTS; i += 1000)
	{
		gettimeofday(&tvBegin, NULL);//linux wall time
		cSection(atoms, N, 4, Qx, Qy, Qz, i, results, magCellSize);
		gettimeofday(&tvEnd, NULL);//linux wall time
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		if(i == 149000)//test
		{	
			for(j = 0; j < i; j++)
			{
				printf("Q: %f   S: %f\n", Qx[j], results[j]);
			}
		}
		fprintf(file,"%d\t%f\n",i,tvDiff.tv_sec*1000 + tvDiff.tv_usec/1000.0); /*writes*/
		printf("Time to do %d points: sec: %d  usec: %d\n", i,tvDiff.tv_sec, tvDiff.tv_usec);
	}
}


//Timing test with (completely arbitrary) 8 atom unit cell
//this should give some useless calculations, but give me an estimate of
//calculation time.
void test5()
{
	#define N 8
	Complex M[N*N];
	int magCellSize[3] = {2,1,1};
	Real *mem = (Real *) malloc((6*N*N + 9*N)*sizeof(Real));

	//linux wall time
	struct timeval tvBegin, tvEnd, tvDiff;
	FILE *file; 

	//Test GPU
	int i, j;
	#define NUMPOINTS 150000
	Real Qx[NUMPOINTS];
	Real Qy[NUMPOINTS];
	Real Qz[NUMPOINTS];
	Real results[NUMPOINTS];
	for(i = 0; i < NUMPOINTS; i++)
	{
		Qx[i] = 0.000033*i;
		Qy[i] = 0.0;
		Qz[i] = 0.0;
	}
	printf("successfully allocated Q and result arrays.\n");

	Atom atoms[8];
	int nbrs1[3] = {1,2,4};
	int nbrs2[2] = {0,3};
	int nbrs3[1] = {0};
	int nbrs4[1] = {1};
	int nbrs5[2] = {0,5};
	int nbrs6[2] = {4, 7};
	int nbrs7[1] = {7};
	int nbrs8[2] = {6, 5};
    Real int1[3] = {-1.0,1.0,-1.0};
	Real int2[2] = {-1.0,-1.0};
	Real int3[1] = {-1.0};
	Real int4[1] = {1.0};
    Real int5[2] = {-1.0, 1.0};
	Real int6[2] = {1.0, 0.5};
	Real int7[1] = {-1.0};
	Real int8[2] = {-1.0, 0.5};
	
	atoms[0].numNeighbors = 3;
	atoms[0].neighbors = nbrs1;
    atoms[0].interactions = int1;
    atoms[0].S = 1.0;
    atoms[0].sigma = 1;
    atoms[0].pos[0] = 0.5;
    atoms[0].pos[1] = 0.5;
    atoms[0].pos[2] = 0.5;
    atoms[0].l[0] = 0;
    atoms[0].l[1] = 0;
    atoms[0].l[2] = 0;
    atoms[0].d[0] = atoms[0].pos[0]/magCellSize[0];
    atoms[0].d[1] = atoms[0].pos[1]/magCellSize[1];
    atoms[0].d[2] = atoms[0].pos[2]/magCellSize[2];

	atoms[1].numNeighbors = 2;
	atoms[1].neighbors = nbrs2;
    atoms[1].interactions = int2;
    atoms[1].S = 1.0;
    atoms[1].sigma = -1;
    atoms[1].pos[0] = 1.5;
    atoms[1].pos[1] = 0.5;
    atoms[1].pos[2] = 0.5;
    atoms[1].l[0] = 0;
    atoms[1].l[1] = 0;
    atoms[1].l[2] = 0;
    atoms[1].d[0] = atoms[1].pos[0]/magCellSize[0];
    atoms[1].d[1] = atoms[1].pos[1]/magCellSize[1];
    atoms[1].d[2] = atoms[1].pos[2]/magCellSize[2];
    
	atoms[2].numNeighbors = 1;
	atoms[2].neighbors = nbrs3;
    atoms[2].interactions = int3;
    atoms[2].S = 1.0;
    atoms[2].sigma = -1;
    atoms[2].pos[0] = -0.5;//Tested and working 5/11/12
    atoms[2].pos[1] = 0.5;
    atoms[2].pos[2] = 0.5;
    atoms[2].l[0] = -1;
    atoms[2].l[1] = 0;
    atoms[2].l[2] = 0;
    atoms[2].d[0] = atoms[2].pos[0]/magCellSize[0];
    atoms[2].d[1] = atoms[2].pos[1]/magCellSize[1];
    atoms[2].d[2] = atoms[2].pos[2]/magCellSize[2];

    atoms[3].numNeighbors = 1;
	atoms[3].neighbors = nbrs4;
    atoms[3].interactions = int4;
    atoms[3].S = 1.0;
    atoms[3].sigma = 1;
    atoms[3].pos[0] = 2.5;
    atoms[3].pos[1] = 0.5;
    atoms[3].pos[2] = 0.5;
    atoms[3].l[0] = 1;
    atoms[3].l[1] = 0;
    atoms[3].l[2] = 0;
    atoms[3].d[0] = atoms[3].pos[0]/magCellSize[0];
    atoms[3].d[1] = atoms[3].pos[1]/magCellSize[1];
    atoms[3].d[2] = atoms[3].pos[2]/magCellSize[2];

	atoms[4].numNeighbors = 2;
	atoms[4].neighbors = nbrs5;
    atoms[4].interactions = int5;
    atoms[4].S = 1.0;
    atoms[4].sigma = 1;
    atoms[4].pos[0] = 0.75;
    atoms[4].pos[1] = 0.5;
    atoms[4].pos[2] = 0.65;
    atoms[4].l[0] = 0;
    atoms[4].l[1] = 0;
    atoms[4].l[2] = 0;
    atoms[4].d[0] = atoms[0].pos[0]/magCellSize[0];
    atoms[4].d[1] = atoms[0].pos[1]/magCellSize[1];
    atoms[4].d[2] = atoms[0].pos[2]/magCellSize[2];

	atoms[5].numNeighbors = 2;
	atoms[5].neighbors = nbrs6;
    atoms[5].interactions = int6;
    atoms[5].S = 1.0;
    atoms[5].sigma = -1;
    atoms[5].pos[0] = 1.5;
    atoms[5].pos[1] = 0.85;
    atoms[5].pos[2] = 1.5;
    atoms[5].l[0] = 0;
    atoms[5].l[1] = 0;
    atoms[5].l[2] = 0;
    atoms[5].d[0] = atoms[1].pos[0]/magCellSize[0];
    atoms[5].d[1] = atoms[1].pos[1]/magCellSize[1];
    atoms[5].d[2] = atoms[1].pos[2]/magCellSize[2];
    
	atoms[6].numNeighbors = 1;
	atoms[6].neighbors = nbrs7;
    atoms[6].interactions = int7;
    atoms[6].S = 1.0;
    atoms[6].sigma = -1;
    atoms[6].pos[0] = -0.5;//Tested and working 5/11/12
    atoms[6].pos[1] = 0.25;
    atoms[6].pos[2] = 0.95;
    atoms[6].l[0] = -1;
    atoms[6].l[1] = 0;
    atoms[6].l[2] = 0;
    atoms[6].d[0] = atoms[2].pos[0]/magCellSize[0];
    atoms[6].d[1] = atoms[2].pos[1]/magCellSize[1];
    atoms[6].d[2] = atoms[2].pos[2]/magCellSize[2];

    atoms[7].numNeighbors = 2;
	atoms[7].neighbors = nbrs8;
    atoms[7].interactions = int8;
    atoms[7].S = 1.0;
    atoms[7].sigma = 1;
    atoms[7].pos[0] = 2.5;
    atoms[7].pos[1] = 1.5;
    atoms[7].pos[2] = 0.35;
    atoms[7].l[0] = 1;
    atoms[7].l[1] = 0;
    atoms[7].l[2] = 0;
    atoms[7].d[0] = atoms[3].pos[0]/magCellSize[0];
    atoms[7].d[1] = atoms[3].pos[1]/magCellSize[1];
    atoms[7].d[2] = atoms[3].pos[2]/magCellSize[2];

//	cSection(atoms, N, 4, Qx, Qy, Qz, NUMPOINTS, results, magCellSize);

	//Timing test
	file = fopen("big_times2.txt","w");
	for(i = 1000; i < NUMPOINTS; i += 1000)
	{
		gettimeofday(&tvBegin, NULL);//linux wall time
		cSection(atoms, N, 8, Qx, Qy, Qz, i, results, magCellSize);
		gettimeofday(&tvEnd, NULL);//linux wall time
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		if(i == 149000)//test
		{	
			for(j = 0; j < i; j++)
			{
				printf("Q: %f   S: %f\n", Qx[j], results[j]);
			}
		}
		fprintf(file,"%d\t%f\n",i,tvDiff.tv_sec*1000 + tvDiff.tv_usec/1000.0); /*writes*/
		printf("Time to do %d points: sec: %d  usec: %d\n", i,tvDiff.tv_sec, tvDiff.tv_usec);
	}
}

//Timing test with 8 atoms in the unit cell - all in an antiferromagnetic chain
//I think the last test produced matrices whose eigendecompositions weren't
//converging, so it was tkaing a particularly long time.
void test6()
{
	#define N 8
	Complex M[N*N];
	int magCellSize[3] = {2,1,1};
	Real *mem = (Real *) malloc((6*N*N + 9*N)*sizeof(Real));

	//linux wall time
	struct timeval tvBegin, tvEnd, tvDiff;
	FILE *file; 

	//Test GPU
	int i, j;
	#define NUMPOINTS 110000
	Real Qx[NUMPOINTS];
	Real Qy[NUMPOINTS];
	Real Qz[NUMPOINTS];
	Real results[NUMPOINTS];
	for(i = 0; i < NUMPOINTS; i++)
	{
		Qx[i] = 0.000033*i;
		Qy[i] = 0.0;
		Qz[i] = 0.0;
	}
	printf("successfully allocated Q and result arrays.\n");

	Atom atoms[8];
	int nbrs1[1] = {1};
	int nbrs2[1] = {0};
	int nbrs3[1] = {3};
	int nbrs4[1] = {2};
	int nbrs5[1] = {5};
	int nbrs6[1] = {4};
	int nbrs7[1] = {7};
	int nbrs8[1] = {6};
    Real int1[1] = {-1.0};
	Real int2[1] = {-1.0};
	Real int3[1] = {-1.0};
	Real int4[1] = {-1.0};
    Real int5[1] = {-1.0};
	Real int6[1] = {-1.0};
	Real int7[1] = {-1.0};
	Real int8[1] = {-1.0};
	
	atoms[0].numNeighbors = 1;
	atoms[0].neighbors = nbrs1;
    atoms[0].interactions = int1;
    atoms[0].S = 1.0;
    atoms[0].sigma = 1;
    atoms[0].pos[0] = 0.5;
    atoms[0].pos[1] = 0.15;
    atoms[0].pos[2] = 0.5;
    atoms[0].l[0] = 0;
    atoms[0].l[1] = 0;
    atoms[0].l[2] = 0;
    atoms[0].d[0] = atoms[0].pos[0]/magCellSize[0];
    atoms[0].d[1] = atoms[0].pos[1]/magCellSize[1];
    atoms[0].d[2] = atoms[0].pos[2]/magCellSize[2];

	atoms[1].numNeighbors = 1;
	atoms[1].neighbors = nbrs2;
    atoms[1].interactions = int2;
    atoms[1].S = 1.0;
    atoms[1].sigma = -1;
    atoms[1].pos[0] = 1.5;
    atoms[1].pos[1] = 0.15;
    atoms[1].pos[2] = 0.5;
    atoms[1].l[0] = 1;
    atoms[1].l[1] = 0;
    atoms[1].l[2] = 0;
    atoms[1].d[0] = atoms[1].pos[0]/magCellSize[0];
    atoms[1].d[1] = atoms[1].pos[1]/magCellSize[1];
    atoms[1].d[2] = atoms[1].pos[2]/magCellSize[2];
    
	atoms[2].numNeighbors = 1;
	atoms[2].neighbors = nbrs3;
    atoms[2].interactions = int3;
    atoms[2].S = 1.0;
    atoms[2].sigma = -1;
    atoms[2].pos[0] = 0.5;//Tested and working 5/11/12
    atoms[2].pos[1] = 0.25;
    atoms[2].pos[2] = 0.5;
    atoms[2].l[0] = 0;
    atoms[2].l[1] = 0;
    atoms[2].l[2] = 0;
    atoms[2].d[0] = atoms[2].pos[0]/magCellSize[0];
    atoms[2].d[1] = atoms[2].pos[1]/magCellSize[1];
    atoms[2].d[2] = atoms[2].pos[2]/magCellSize[2];

    atoms[3].numNeighbors = 1;
	atoms[3].neighbors = nbrs4;
    atoms[3].interactions = int4;
    atoms[3].S = 1.0;
    atoms[3].sigma = 1;
    atoms[3].pos[0] = 1.5;
    atoms[3].pos[1] = 0.25;
    atoms[3].pos[2] = 0.5;
    atoms[3].l[0] = 1;
    atoms[3].l[1] = 0;
    atoms[3].l[2] = 0;
    atoms[3].d[0] = atoms[3].pos[0]/magCellSize[0];
    atoms[3].d[1] = atoms[3].pos[1]/magCellSize[1];
    atoms[3].d[2] = atoms[3].pos[2]/magCellSize[2];

	atoms[4].numNeighbors = 1;
	atoms[4].neighbors = nbrs5;
    atoms[4].interactions = int5;
    atoms[4].S = 1.0;
    atoms[4].sigma = 1;
    atoms[4].pos[0] = 0.5;
    atoms[4].pos[1] = 0.35;
    atoms[4].pos[2] = 0.5;
    atoms[4].l[0] = 0;
    atoms[4].l[1] = 0;
    atoms[4].l[2] = 0;
    atoms[4].d[0] = atoms[0].pos[0]/magCellSize[0];
    atoms[4].d[1] = atoms[0].pos[1]/magCellSize[1];
    atoms[4].d[2] = atoms[0].pos[2]/magCellSize[2];

	atoms[5].numNeighbors = 1;
	atoms[5].neighbors = nbrs6;
    atoms[5].interactions = int6;
    atoms[5].S = 1.0;
    atoms[5].sigma = -1;
    atoms[5].pos[0] = 1.5;
    atoms[5].pos[1] = 0.35;
    atoms[5].pos[2] = 0.5;
    atoms[5].l[0] = 1;
    atoms[5].l[1] = 0;
    atoms[5].l[2] = 0;
    atoms[5].d[0] = atoms[1].pos[0]/magCellSize[0];
    atoms[5].d[1] = atoms[1].pos[1]/magCellSize[1];
    atoms[5].d[2] = atoms[1].pos[2]/magCellSize[2];
    
	atoms[6].numNeighbors = 1;
	atoms[6].neighbors = nbrs7;
    atoms[6].interactions = int7;
    atoms[6].S = 1.0;
    atoms[6].sigma = -1;
    atoms[6].pos[0] = 0.5;
    atoms[6].pos[1] = 0.45;
    atoms[6].pos[2] = 0.5;
    atoms[6].l[0] = 0;
    atoms[6].l[1] = 0;
    atoms[6].l[2] = 0;
    atoms[6].d[0] = atoms[2].pos[0]/magCellSize[0];
    atoms[6].d[1] = atoms[2].pos[1]/magCellSize[1];
    atoms[6].d[2] = atoms[2].pos[2]/magCellSize[2];

    atoms[7].numNeighbors = 1;
	atoms[7].neighbors = nbrs8;
    atoms[7].interactions = int8;
    atoms[7].S = 1.0;
    atoms[7].sigma = 1;
    atoms[7].pos[0] = 1.5;
    atoms[7].pos[1] = 0.45;
    atoms[7].pos[2] = 0.5;
    atoms[7].l[0] = 1;
    atoms[7].l[1] = 0;
    atoms[7].l[2] = 0;
    atoms[7].d[0] = atoms[3].pos[0]/magCellSize[0];
    atoms[7].d[1] = atoms[3].pos[1]/magCellSize[1];
    atoms[7].d[2] = atoms[3].pos[2]/magCellSize[2];

//	cSection(atoms, N, 4, Qx, Qy, Qz, NUMPOINTS, results, magCellSize);

	//Timing test
	file = fopen("big2_times2.txt","w");
	for(i = 1000; i < NUMPOINTS; i += 1000)
	{
		gettimeofday(&tvBegin, NULL);//linux wall time
		cSection(atoms, N, 8, Qx, Qy, Qz, i, results, magCellSize);
		gettimeofday(&tvEnd, NULL);//linux wall time
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		if(i == 109000)//test
		{	
			for(j = 0; j < i; j++)
			{
				printf("Q: %f   S: %f\n", Qx[j], results[j]);
			}
		}
		fprintf(file,"%d\t%f\n",i,tvDiff.tv_sec*1000 + tvDiff.tv_usec/1000.0); /*writes*/
		printf("Time to do %d points: sec: %d  usec: %d\n", i,tvDiff.tv_sec, tvDiff.tv_usec);
	}
}


int main()
{
	test6();
}
