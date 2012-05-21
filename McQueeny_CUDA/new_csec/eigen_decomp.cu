#include <malloc.h>
#include "book.h"

/*
Note:
Matrices are one dimensional arrays which are indexed using the macro INDEX.

How this works:
First the matrix is converted to upper Hessenberg form using householder
transformations.  Then the QR algorithm is used to produce an upper triangular
matrix from which the eigenvalues can be read off the diagonal.  The function
eigenDecomp performs all of these steps, producing the eigenvalues and right
eigenvectors.

Most of the code was written based on pseudocode in Matrix Algorithms Vol. 2 by
G.W. Stewart.


This was first written to run on a CPU and then converted to use CUDA.  It uses
a lot of device main memory and could probably be tweeked a lot to improve speed
on the GPU.
*/

#include <stdio.h>

//---------------------- Helper Functions ----------------------------

__host__ Complex host_complexMult(Complex a, Complex b)
{
	Complex result;
	result.real = a.real*b.real - a.imag*b.imag;
	result.imag = a.imag*b.real + a.real*b.imag;
	return result;
}


__device__ void copyVect(Complex *src, Complex *dest, int n)
{
	int i;
	for(i = 0; i < n; i++)
	{
		dest[i] = src[i];
	}
}


__device__ Real norm(Complex *vect, int n)
{
	int i;
	float norm;
	norm = 0.0;
	for(i = 0; i < n; i++)
	{
		norm += vect[i].real*vect[i].real + vect[i].imag*vect[i].imag;
	}
	return sqrt(norm);
}

__host__ Real host_norm(Complex *vect, int n)
{
	int i;
	float norm;
	norm = 0.0;
	for(i = 0; i < n; i++)
	{
		norm += vect[i].real*vect[i].real + vect[i].imag*vect[i].imag;
	}
	return sqrt(norm);
}


__device__ void copyMat(Complex *src, Complex *dest, int dim1, int dim2)
{
	copyVect(src, dest, dim1*dim2);
}

/*
Multiply matrices mat1 and mat2 and put the result into dest.  mat1 is an n1*m1 matrix.
mat2 is an m1*m2 matrix and dest is an n1*m2 matrix.
*/
__device__ void matrixMult(Complex *mat1, int n1, int m1, Complex *mat2, int m2, Complex *dest)
{
	int i, j, k;
	Complex tmp;
	
	//first fill dest with zeros
	for(i = 0; i < n1; i++)
	{
		for(j = 0; j < m2; j++)
		{
			INDEX(dest, n1, i, j).real = 0.0;
			INDEX(dest, n1, i, j).imag = 0.0;
		}
	}
	
	//now do the multiply
	for(i = 0; i < n1; i++)
	{
		for(j = 0; j < m1; j++)
		{
			for(k = 0; k < m2; k++)
			{
				tmp = complexMult(INDEX(mat1, n1, i, j), INDEX(mat2, m1, j, k));
				INDEX(dest, n1, i, k).real += tmp.real;
				INDEX(dest, n1, i, k).imag += tmp.imag;
			}
		}
	}
}


//--------------------- EigenDecomposition Functions -------------------------
/*
First the matrix is converted to upper Hessenberg form using householder
transformations.  Then the QR algorithm is used to produce an upper triangular
matrix from which the eigenvalues can be read off the diagonal.  The function
eigenDecomp performs all fo these steps, producing the eigenvalues and right
eigenvectors.
*/


/*
Algorithm taken from Matrix Algorithms Vol. II by G.W. Stewart.
His original pseudocode is in the comments.

This function takes a vector a and produces a u that generates the
Householder transformation H = I - uuH such that Ha = ve1.

n is the length of the vectors.
*/
__device__ void housegen(Complex *a, Complex *u, Complex *v, int n)
{
	int i;
	Real tmp;
	Complex rho;
	
	//u = a
	copyVect(a,u,n);
	
	//v = ||a||2
	v->real = norm(a,n);
	v->imag = 0.0;//must be done after norm is taken above!
	
	//if(v = 0)u[1] = sqrt(2); return; fi  <-no clue what 'fi' is
	if(v->real == 0.0)//v can only be real right now   should i be using epsilon?
	{
		u[0].real = sqrt(2.0);
		u[0].imag = 0;
		return;
	}
	
	//if(u[1] != 0)
	if(u[0].real != 0.0 || u[0].imag != 0.0)// not sure if I should use an epsilon here?
	{
		//rho = u_bar[1]/|u[1]|
		tmp = ABS(u[0]);
		rho.real = u[0].real/tmp;
		rho.imag = -u[0].imag/tmp;
	}
	else//else rho = 1;
	{
		rho.real = 1.0;
		rho.imag = 0.0;
	}
		
	//u = (rho/v)*u
	for(i = 0; i < n; i++)
	{
		//v can still only be real at this point
		u[i] = complexMult(rho, u[i]);
		u[i].real /= v->real;
		u[i].imag /= v->real;
	}
		
	//u[1] = 1 + u[1]
	u[0].real = 1.0 + u[0].real;
	
	
	//u = u/sqrt(u[1])
	tmp = u[0].real;
	for(i = 0; i < n; i++)
	{
		u[i].real = u[i].real/sqrt(tmp);//I think u[0] has tpo be real here
		u[i].imag = u[i].imag/sqrt(tmp);
	}
	
	//v = -rho_bar*v
	rho.real = -rho.real;
	*v = complexMult(rho, *v);
}


/*
Algorithm taken from Matrix Algorithms Vol. II by G.W. Stewart.

This function takes a vector A of order n  and reduces it to Hessenberg form
by Householder transformations.

In the future it may be possible to remove H entirely and do the transformation
in place.

A is an n*n matrix that I am transforming
H is an n*n matrix where I will put the result
Q is an n*n matrix where I will accumulate the transformations
u is a vector of length n for scratch work.
vH is another vector of length n for scratch work
*/
__device__ void hessreduce(Complex *A, Complex *H, Complex *Q, Complex *u, Complex *vH, int n)
{
	int k, i, j;
	Complex tmp;
	
	if(n < 2)
		return;
	
	//Reduce A
	//H = A
	copyMat(A,H,n,n);
	
	//3. for k = 1 to n-2
	for(k = 0; k < n-2; k++)
	{
		//Generate the Transformation
		//housegen(H[k+1:n,k],u,H[k+1,k])
		housegen(&INDEX(H,n,(k+1),k),u+k+1,&INDEX(H,n,(k+1),k),n-k-1);
		
		//5. Q[k+1:n,k] = u
		copyVect(u+k+1, &INDEX(Q, n, (k+1), k), n-k-1);
		
		//Premultiply the transformation
		//6. vH = uH*H[k+1:n,k+1:n]
		for(i = k+1; i < n; i++)
		{
			vH[i].real = 0.0;
			vH[i].imag = 0.0;
			
			for(j = k+1; j < n; j++)
			{
				tmp = INDEX(H,n,j,i);
				vH[i].real += u[j].real * tmp.real;
				vH[i].real += u[j].imag * tmp.imag;//minus minus for hermitian
				vH[i].imag += u[j].real * tmp.imag;
				vH[i].imag -= u[j].imag * tmp.real;//minus sign is for hermitian
			}
		}
		
		//7. H[k+1:n,k+1:n] = H[k+1:n, k+1:n] - u*vH
		for(i = k+1; i < n; i++)
		{
			for(j = k+1; j < n; j++)
			{
				INDEX(H,n,i,j).real -= u[i].real *vH[j].real;
				INDEX(H,n,i,j).real += u[i].imag *vH[j].imag;
				INDEX(H,n,i,j).imag -= u[i].real *vH[j].imag;
				INDEX(H,n,i,j).imag -= u[i].imag *vH[j].real;
			}
		}
						
		//H[k+2:n,k] = 0
		for(i = k+2; i < n; i++)
		{
			INDEX(H,n,i,k).real = 0.0;
			INDEX(H,n,i,k).imag = 0.0;
		}
		
		//Postmultiply the transformation
		//9. v = H[1:n, k+1:n]*u
		//I will use the variable vH for v (for space).
		for(i = 0; i < n; i++)
		{
			vH[i].real = 0.0;
			vH[i].imag = 0.0;
			for(j = k+1; j < n; j++)
			{
				tmp = INDEX(H,n,i,j);
				vH[i].real += tmp.real * u[j].real;
				vH[i].real -= tmp.imag * u[j].imag;
				vH[i].imag += tmp.real * u[j].imag;
				vH[i].imag += tmp.imag * u[j].real;
			}
		}
		
		//10. H[1:n, k+1:n] = H[1:n,k+1:n] - v*uH
		for(i = 0; i < n; i++)
		{
			for(j = k+1; j < n; j++)
			{
				INDEX(H,n,i,j).real -= vH[i].real * u[j].real;
				INDEX(H,n,i,j).real -= vH[i].imag * u[j].imag;
				INDEX(H,n,i,j).imag += vH[i].real * u[j].imag;
				INDEX(H,n,i,j).imag -= vH[i].imag * u[j].real;
			}
		}
	}//end k
	
	//Accumulate the Transformations
	//12. Q[:,n] = e_n; Q[:,n-1] = e_(n-1)
	for(i = 0; i < n; i++)
	{
		INDEX(Q,n,i,(n-1)).real = 0.0;
		INDEX(Q,n,i,(n-1)).imag = 0.0;
		INDEX(Q,n,i,(n-2)).real = 0.0;
		INDEX(Q,n,i,(n-2)).imag = 0.0;
	}
	INDEX(Q,n,(n-1),(n-1)).real = 1.0;
	INDEX(Q,n,(n-2),(n-2)).real = 1.0;
	
	//13. for k = n-2 to 1 by -1
	for(k = n-3; k >= 0; k--)
	{	
		//14. u = Q[k+1:n,k]
		copyVect(&INDEX(Q,n,(k+1),k), u+k+1, n-(k+1));
		
		//15. vH = uH*Q[k+1:n,k+1:n]//Q[k+1:n,k] = u
		for(i = k+1; i < n; i++)
		{
			vH[i].real = 0.0;
			vH[i].imag = 0.0;
			for(j = k+1; j < n; j++)
			{
				tmp.real = u[j].real;
				tmp.imag = -u[j].imag;
				tmp = complexMult(tmp, INDEX(Q, n, j, i));
				vH[i].real += tmp.real;
				vH[i].imag += tmp.imag;
			}
		}
		
		//16. Q[k+1:n, k+1:n] = Q[k+1:n, k+1:n] -u*vH
		for(i = k+1; i < n; i++)
		{
			for(j = k+1; j < n; j++)
			{
				tmp = complexMult(u[i],vH[j]);
				INDEX(Q,n,i,j).real -= tmp.real;
				INDEX(Q,n,i,j).imag -= tmp.imag;	
			}
		}
		
		//17. Q[:,k] = e_k
		for(i = 0; i < n; i++)
		{
			INDEX(Q,n,i,k).real = 0.0;
			INDEX(Q,n,i,k).imag = 0.0;
		}
		INDEX(Q,n,k,k).real = 1.0;
		
	}// 18. end for k
}//end hessreduce


/*
Algorithm taken from Matrix Algorithms Vol. II by G.W. Stewart.

This function creates a plane rotation from a and b.  The cosine of the rotation
is real.  a is overwritten with its final value and b is overwritten with zero.

TODO:
Right now a and b are being compared with 0.0.  Depending on how they are
computed before being passed in, I may want to cmpare with an epsilon instead.
*/
__device__ void rotgen(Complex *a, Complex *b, Real *c, Complex *s)
{
	Complex u;
	Real v, norm_a, norm_b_sq;
	
	//2. if(b = 0)
	if(b->real == 0.0 && b->imag == 0.0)
	{
		//3. c = 1;s=0;return
		*c = 1.0;
		s->real = 0.0;
		s->imag = 0.0;
		return;
	}//4. endif
	
	//5. If (a = 0)
	if(a->real == 0.0 && a->imag == 0.0)
	{
		//6. c = 0; s = 1; a = b; b = 0;return
		*c = 0.0;
		s->real = 0.0;
		s->imag = 0.0;
		a->real = b->real;
		a->imag = b->imag;
		b->real = 0.0;
		b->imag = 0.0;
		return;
	}//7. endif
	
	//8. mu = a/|a|
	norm_a = sqrt(a->real * a->real + a->imag * a->imag);
	u.real = a->real/norm_a;
	u.imag = a->imag/norm_a;
	
	//9. tau = |a| + |b|
	//They instroduced this becuase they said it would save us a sqrt
	//operation, claiming that t = |Re(a)| + |Im(a)| + |Re(b)| + |Im(b)|,
	//which I don't see.
	
	//10. v = t*sqrt(|a/t|^2 + |b/t|^2)
	//Am I mistaken or does this not equal sqrt(|a|^2 + |b|^2),since t is real?
	//It is
	norm_b_sq = b->real * b->real + b->imag * b->imag;
	v = sqrt(norm_a*norm_a + norm_b_sq);
	
	//11. c = |a|/v
	*c = norm_a/v;
	
	//12. s = u*b_bar/v
	b->imag = -b->imag;//It's okay that I'm overwriting this - not used anymore
	*s = complexMult(u,*b);
	s->real = s->real/v;
	s->imag = s->imag/v;
	
	
	//13. a = v*u
	a->real = v*u.real;
	a->imag = v*u.imag;
	
	//14. b = 0
	b->real = 0.0;
	b->imag = 0.0;
	
}//15. end rotgen	
	
	
/*
Algorithm taken from Matrix Algorithms Vol. II by G.W. Stewart.

This function takes a plane otation, P, specified by c and s and two vectors,
x and y and overwrites them with:

 ( xT )
P(    )
 ( yT )
 
The T is a superscript for transpose.

t is a temporary array which is the same size as x and y
n is the length of the vectors - it seems from rotgen that n should always be 2,
but I'm not sure it's used that way.
*/
__device__ void rotapp(Real *c, Complex *s, Complex *x, Complex *y, Complex *t, int n)
{
	int i;
	
	//2. t = c*x + s*y
	for(i = 0; i < n; i++)
	{
		t[i].real = (*c)*x[i].real + s->real*y[i].real - s->imag*y[i].imag;
		t[i].imag = (*c)*x[i].imag + s->real*y[i].imag + s->imag*y[i].real;
	}

	//3. y = c*y -s_bar*x
	for(i = 0; i < n; i++)
	{
		y[i].real = (*c)*y[i].real - s->real*x[i].real - s->imag*x[i].imag;
		y[i].imag = (*c)*y[i].imag - s->real*x[i].imag + s->imag*x[i].real;
	}

	//4. x = t
	for(i = 0; i < n; i++)
	{
		x[i] = t[i];
	}

}//5. end rotapp



/*
Algorithm taken from Matrix Algorithms Vol. II by G.W. Stewart.

This function takes a Hessenberg matrix, H, of order n and an index, l, and
determines indices i1,i2 <= l such that one of the following two conditions
holds:

1. 1 <= i1 < i2 <= l, in which case the matrix deflates at rows i1 and i2
2. 1 = i1 = i2, in which case all the subdiagonal elements of H in rows 1
through l are zero.

H is an n*n matrix

*/
__device__ void backsearch(Complex *H, int n, int l, int *i1, int *i2, Real epsilon)
{
	//2. i1 = i2 = l
	*i1 = l;
	*i2 = l;
	
	//3. while(i1 > 1)
	while(*i1 > 0)
	{
		//4. if(H[i1,i1-1] is negligible)
		if(INDEX(H,n,*i1,(*i1-1)).real * INDEX(H,n,*i1,(*i1-1)).real < epsilon && INDEX(H,n,*i1,(*i1-1)).imag * INDEX(H,n,*i1,(*i1-1)).imag < epsilon)
		{
			//5. H[i1, i1-1] = 0
			INDEX(H,n,*i1,(*i1-1)).real = 0.0;
			INDEX(H,n,*i1,(*i1-1)).imag = 0.0;
		
			//6. if(i1 = i2)
			if(*i1 == *i2)
			{
				//7. i2 = i1 = i1-1
				*i1 = *i1-1;
				*i2 = *i1;
			}
			else//8. else
			{
				//9. leave while
				return;
			}//10. endif
		}
		else//11. else
		{
			//12. i1 = i1-1
			*i1 = *i1-1;
		}//13. endif
	}//14. end while
}//15. end backsearch
	

/*
Algorithm taken from Matrix Algorithms Vol. II by G.W. Stewart.

Given the elements of the matrix:

B = | a  b |
	| c  d |
	
wilkshift returns the eigenvalue of B that is nearest to d
*/
__device__ void wilkshift(Complex *a, Complex *b, Complex *c, Complex *d, Complex *k)
{
	Real s, tmp;
	Complex q, p, r, c_tmp;

	//2. k = d
	k->real = d->real;
	k->imag = d->imag;
	
	//3. s = |a| + |b| + |c| + |d|
	s = sqrt(a->real*a->real + a->imag*a->imag) +
		sqrt(b->real*b->real + b->imag*b->imag) +
		sqrt(c->real*c->real + c->imag*c->imag) +
		sqrt(d->real*d->real + d->imag*d->imag);
		
	//4. if(s=0) return fi
	if(s == 0.0)//TODO - This should probably be an epsilon
		return;
	
	//5. q = (b/s)*(c/s)
	q = complexMult(*b,*c);
	q.real = q.real/s;
	q.imag = q.imag/s;
	q.real = q.real/s;
	q.imag = q.imag/s;
	
	//6. if(q != 0)
	if(q.real != 0.0 || q.imag != 0.0)//TODO - again, epsilon?
	{
		//7. p = 0.5*((a/s) - (d/s))
		p.real = 0.5*(a->real/s - d->real/s);
		p.imag = 0.5*(a->imag/s - d->imag/s);
		
		//8. r = sqrt(p^2 + q)
		//According to wikipedia, the principal square root of a complex number
		//in cartesian coordinates is:
		//sqrt(x + iy) = sqrt((r+x)/2) +/- i*sqrt((r-x)/2)
		//where r = |x+iy| = sqrt(x^2 + y^2)
		//The other root is -1 times the principal root (naturally)
		
		c_tmp = complexMult(p,p);
		c_tmp.real += q.real;
		c_tmp.imag += q.imag;//c_tmp is now p^2 + q
		
		tmp = sqrt(c_tmp.real*c_tmp.real + c_tmp.imag*c_tmp.imag);
		
		r.real = sqrt((tmp + c_tmp.real)/2.0);
		r.imag = sqrt((tmp - c_tmp.real)/2.0);
		
		//9. if(Re(p)*Re(r) + Im(p)*Im(r) < 0) r = -r fi
		if(p.real*r.real + p.imag*r.imag < 0.0)
			r.real = -r.real;
			r.imag = -r.imag;
			
		//10. k = k - s*(q/(p+r))
		c_tmp.real = p.real + r.real;
		c_tmp.imag = p.imag + r.imag;
		tmp = c_tmp.real*c_tmp.real + c_tmp.imag*c_tmp.imag;
		c_tmp.real = c_tmp.real/tmp;
		c_tmp.imag = -c_tmp.imag/tmp;
		//now c_tmp = 1/(p+r)
		c_tmp = complexMult(q,c_tmp);

		k->real -= s*c_tmp.real;
		k->imag -= s*c_tmp.imag;
	}//11. end if
}//12. end wilkshift


/*
Algorithm taken from Matrix Algorithms Vol. II by G.W. Stewart.

Given an upper Hessenberg matrix, H, hqr overwrites it iwth a unitary similar
triangular matrix whose diagonals are the eigenvalues of H.

I beleive this is called Schur form.

n is the size of the matrix H

-1 is returned if more than maxiter iterations are required to to deflate
the matrix at any eigenvalue.  If everything completes successfully, a 0 is
returned.

c, s, r1, r2, and t are arrays of length n, used for scratch work

*/
__device__ int hqr(Complex *H, Complex *Q, Real *c, Complex *s, Complex *r1, Complex *r2, Complex *t, int n, int maxiter, Real epsilon)
{
	int i1, i2, iter, oldi2, i, j;
	Complex k, tmp;
	int retries = 0;
	//2. i1 = 1; i2 = n
	i1 = 0;
	i2 = n-1;//this is used both as an index and an upper bound in loops, so
	//it must be n-1, but I must use <= in loops.
	
	//3. iter = 0;
	iter = 0;
	
	//4. while(true)
	while(1)
	{
		//5. iter = iter+1
		iter += 1;
		
		//6. if(iter > maxiter) error return fi
		if(iter > maxiter)
			return -1;
			
		//7. oldi2 = i2
		oldi2 = i2;
		
		//8. backsearch(H, i2, i1, i2)
		backsearch(H, n, i2, &i1, &i2, epsilon);

		//9. if(i2 = 1) leave hqr fi
		if(i2 == 0)
			return 0;
			
		//10. if(i2 != oldi2)iter = 0 fi
		if(i2 != oldi2)
			iter = 0;//I suppose we moved to the next eigenvalue
		
		//11. wilkshift(H[i2-1,i2-1], H[i2-1,i2], H[i2,i2-1], H[i2,i2], k)
		wilkshift(&INDEX(H,n,(i2-1),(i2-1)), &INDEX(H,n,(i2-1),i2), &INDEX(H,n,i2,(i2-1)), &INDEX(H,n,i2,i2), &k);
/*
		//this is an ad hoc shift if we have failed to converge
		//Stewart says this is the shift used by CGEEV in Lapack
		if(iter == maxiter && retries < 2)
		{
			iter = 0;
			retries += 1;
			k.imag = 0.0;
			tmp.real = INDEX(H, n, i2, (i2-1)).real;
			if(tmp.real >= 0.0)
				k.real = tmp.real;
			else
				k.real = -tmp.real;

			if(i2 > 1)
				tmp.real = INDEX(H, n, (i2-1), (i2-2)).real;
			else//This case is not mentioned, but they call it ad hoc, so what the hey
				tmp.real = INDEX(H, n, (i2-1), i2).real;
			if(tmp.real >= 0.0)
				k.real += tmp.real;
			else
				k.real += -tmp.real;
		}
*/				
		
		//12. H[i1,i1] = H[i1,i1] - k
		INDEX(H,n,i1,i1).real -= k.real;
		INDEX(H,n,i1,i1).imag -= k.imag;

		//13. for i = i1 to i2-1
		for(i = i1; i <= i2-1; i++)
		{
			//14. rotgen(H[i,i], H[i+1, i], c_i, s_i)
			rotgen(&INDEX(H,n,i,i), &INDEX(H,n,(i+1),i), &c[i], &s[i]);
			
			//15. H[i+1, i+1] = H[i+1, i+1] - k
			INDEX(H,n,(i+1),(i+1)).real -= k.real;
			INDEX(H,n,(i+1),(i+1)).imag -= k.imag;
			
			//16. rotapp(c_i, s_i, H[i, i+1:n], H[i+1,i+1:n])
			//Unfortunately, we are now using a row.  Before we were looking at
			//single columns, so I indexed the arrays H[i,j] = H[i + j*n], so
			//that &INDEX(H,n,i,j) could be used to equal H[i:n,j].  I can't do
			//that with rows now.
			//I will be using the array r1 and r2 for these two rows
			//copy the contents fo the rows to r1,r2
			for(j = i+1; j < n; j++)
			{
				r1[j].real = INDEX(H,n,i,j).real;
				r1[j].imag = INDEX(H,n,i,j).imag;
				r2[j].real = INDEX(H,n,(i+1),j).real;
				r2[j].imag = INDEX(H,n,(i+1),j).imag;
			}
			
			rotapp(&c[i], &s[i], &r1[i+1], &r2[i+1], t, n-i-1);
			
			//now copy the results back to H
			for(j = i+1; j < n; j++)
			{
				INDEX(H,n,i,j).real = r1[j].real;
				INDEX(H,n,i,j).imag = r1[j].imag;
				INDEX(H,n,(i+1),j).real = r2[j].real;
				INDEX(H,n,(i+1),j).imag = r2[j].imag;
			}
		}//17. end for i

		//18. for i = i1 to i2-1
		for(i = i1; i <= i2-1; i++)
		{
			//19. rotapp(c_i, s_i_bar, H[1:i+1, i], H[1:i+1, i+1]
			tmp.real = s[i].real;
			tmp.imag = -s[i].imag;
			//I can use the column as a continuous array
			rotapp(&c[i], &tmp, &INDEX(H,n,0,i), &INDEX(H,n,0,(i+1)), t, i+2);
			
			//20. rotapp(c_i, s_i_bar, Q[1:n,i], Q[1:n,i+1)
			//I can use the column as a continuous array
			rotapp(&c[i], &tmp, &INDEX(Q,n,0,i), &INDEX(Q,n,0,(i+1)), t, n);
			
			//21. H[i,i] = H[i,i] + k
			INDEX(H,n,i,i).real += k.real;
			INDEX(H,n,i,i).imag += k.imag;
		}//22. end for i

		//23. H[i2,i2] = H[i2,i2] + k
		INDEX(H,n,i2,i2).real += k.real;
		INDEX(H,n,i2,i2).imag += k.imag;
	}//24. end while
}//25 end hqr
					

/*
Algorithm taken from Matrix Algorithms Vol. II by G.W. Stewart.

Returns the right eigenvectors of the upper tiangular matrix T in the matrix X.
T and X are n*n Complex matrices.
*/
__device__ void righteigvec(Complex *T, Complex *X, int n)
{
    int k, i, j;
    Real dmin, tmp, s;
    Complex d, tmp_c;
    
    //fill X with zeros just in case
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
        {
            INDEX(X,n,i,j).real = 0.0;
            INDEX(X,n,i,j).imag = 0.0;
        }
    }
    
	//4. for k = n to 1 by -1
	for(k = n-1; k >= 0; k--)
	{   
	    //5. X[1:k-1,k] = -T[1:k-1,k]
	    for(i = 0; i <= k-1; i++)
	    {
	        INDEX(X,n,i,k).real = -INDEX(T,n,i,k).real;
	        INDEX(X,n,i,k).imag = -INDEX(T,n,i,k).imag;	
	    }
	    
	    //6. X[k,k] = 1
	    INDEX(X,n,k,k).real = 1.0;
	    INDEX(X,n,k,k).imag = 0.0;

	    //7. X[k+1:n,k] = 0
	    for(i = k+1; i < n; i++)
	    {
	        INDEX(X,n,i,k).real = 0.0;
	        INDEX(X,n,i,k).imag = 0.0;	    
	    }
	    
	    //8. dmin = max{eps_M*|T[k,k]|,smallnum}
	    dmin = 0.0;
	    tmp = INDEX(T,n,k,k).real * INDEX(T,n,k,k).real + INDEX(T,n,k,k).imag * INDEX(T,n,k,k).imag;
	    tmp = EPS_M * sqrt(tmp);
	    if(tmp > SMALLNUM)
	        dmin = tmp;
	    else
	        dmin = SMALLNUM;
	    
	    //9. for j = k-1 to 1 by -1
	    for(j = k-1; j >= 0; j--)
	    {
	        //10. d = T[j,j] - T[k,k]
	        d.real = INDEX(T,n,j,j).real - INDEX(T,n,k,k).real;
	        d.imag = INDEX(T,n,j,j).imag - INDEX(T,n,k,k).imag;
	        
	        //11. if(|d| <= dmin) d = dmin fi
	        if(norm(&d,1) <= dmin)
	        {
	            d.real = dmin;
	            d.imag = 0.0;
	        }
	        
	        //12. if(|X[j,k]|/bignum >= |d|)
	        if(norm(&(INDEX(X,n,j,k)),1)/BIGNUM >= norm(&d,1))
	        {
	            //13. s = |d|/|X[j,k]|
	            s = norm(&d, 1)/norm(&INDEX(X,n,j,k),1);
	            
	            //14. X[1:k,k] = s*X[1:k,k]
	            for(i = 0; i <= k; i++)
	            {
	                INDEX(X,n,i,k).real = s*INDEX(X,n,i,k).real;
	                INDEX(X,n,i,k).imag = s*INDEX(X,n,i,k).imag;
	            }
	        }//15. endif
	        
	        //16. X[j,k] = X[j,k]/d
	        tmp = INDEX(X,n,j,k).real;
	        INDEX(X,n,j,k).real = INDEX(X,n,j,k).real*d.real + INDEX(X,n,j,k).imag*d.imag;
	        INDEX(X,n,j,k).imag = INDEX(X,n,j,k).imag*d.real - tmp*d.imag;
	        tmp = d.real*d.real + d.imag*d.imag;
	        INDEX(X,n,j,k).real /= tmp;
	        INDEX(X,n,j,k).imag /= tmp;
	        
	        //17. X[1:j-1,k] = X[1:j-1,k] - X[j,k]*T[1:j-1,j]
	        for(i = 0; i <= j-1; i++)
	        {
	            tmp_c = complexMult(INDEX(X,n,j,k), INDEX(T,n,i,j));
	            INDEX(X,n,i,k).real = INDEX(X,n,i,k).real - tmp_c.real;
	            INDEX(X,n,i,k).imag = INDEX(X,n,i,k).imag - tmp_c.imag;
	        }
	        
	        
	    }//18. end for j
	    
	    //19. X[1:k,k] = X[1:k,k]/||X[1:k,k]||_2
	    tmp = norm(&INDEX(X,n,0,k),k+1);
	    for(i = 0; i <= k; i++)
	    {
	        INDEX(X,n,i,k).real /= tmp;
	        INDEX(X,n,i,k).imag /= tmp;
	    }
	}//20. end for k
}//21. end righteigvec


/*
Perform an eigendecomposition of the complex n*n matrix A.
eigenValues is an array of length n where the eigenvalues will be stored.
V is an n*n matrix where the right eigenvalues will be stored.
rv1,cv1,cv2, and cv3 are arrays of length n used for scratch work.
Any values below espilon will be considered zero.  A good value would probably
be the largest value that could be added to the largest value in the matrix
without changing it at the given Real precision (float or double).
maxiter is the maximum number of times we will attempt to deflate the matrix at
one eigenvalue (10-50 is probably good).
*/
__device__ void eigenDecomp(Complex *A, Complex *V, Complex *Q, Complex *eigenValues, Real *rv1, Complex *cv1, Complex *cv2, Complex *cv3, int n, Real epsilon, int maxiter)
{
	int i, j, k;
	Complex tmp_c;
	
	//Put the upper hessenberg form of A into V
	hessreduce(A, V, Q, cv1, cv2, n);
	
	//Convert V into Schur form
	hqr(V, Q, rv1, cv1, cv2, cv3, eigenValues, n, 50, epsilon);
	
	//read the eigenvalues off the diagonal of the Schur matrix
	for(i = 0; i < n; i++)
	{
		eigenValues[i] = INDEX(V,n,i,i);
	}
	
	//Now if I want to compute the eigenvectors, I can either use the method in
	//the G.W. Stewart book or I can use gaussian elimination on the original
	//matrix.  Both methods would require a third matrix, so I went with his
	//method.
	
	righteigvec(V, A, n);
	
	//now multiply the eigenvectors in X by Q and store in V (V = QX) (X = A)
	for(i = 0; i < n; i++)
	{
	    for(j = 0; j < n; j++)
	    {
	        INDEX(V,n,i,j).real = 0.0;
	        INDEX(V,n,i,j).imag = 0.0;
	        for(k = 0; k < n; k++)
	        {
	            tmp_c = complexMult(INDEX(Q,n,i,k), INDEX(A,n,k,j));
	            INDEX(V,n,i,j).real -= tmp_c.real;
	            INDEX(V,n,i,j).imag -= tmp_c.imag;
	        }
	    }
	}
}



//-------------------------------- Testing -------------------------------------

//this is a global function which calls eigenDecomp (a device function)
__global__ void eigen_caller(Complex *A, Complex *V, Complex *Q, Complex *eigenValues, Real *rv1, Complex *cv1, Complex *cv2, Complex *cv3, int n, Real epsilon, int maxiter)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	eigenDecomp(A+tid*n*n, V+tid*n*n, Q+tid*n*n, eigenValues+tid*n, rv1+tid*n, cv1+tid*n, cv2+tid*n, cv3+tid*n, n, epsilon, maxiter);
}

//This function takes the matrix A and a vector l containing its eigenvalues
//as well as a matrix V containing the eigenvectors.  For each vector v in V
//it finds A*v-l*v and records the norm.  Ideally it should be
//zero.  The lagest norm will be returned.
//This function uses malloc, so it can't be used on the GPU.
Real eigenTest(Complex *A, Complex *V, Complex *l, int n)
{
	int i,j,k;
	Complex tmp_c;
	Real max = -1.0;
	Complex *tmp = (Complex *)malloc(n*sizeof(Complex));

	for(i = 0; i < n; i++)//this will loop through eigenvectors
	{
	    for(j = 0; j < n; j++)
	    {
	        tmp[j].real = 0.0;
	        tmp[j].imag = 0.0;
	        for(k = 0; k < n; k++)
	        {
	            tmp_c = host_complexMult(INDEX(A,n,j,k), INDEX(V,n,k,i));
	            tmp[j].real += tmp_c.real;
	            tmp[j].imag += tmp_c.imag;
	        }
	    }
	    
	    for(j = 0; j < n; j++)
	    {
	    	tmp_c = host_complexMult(INDEX(V,n,j,i), l[i]);
	    	tmp[j].real -= tmp_c.real;
	    	tmp[j].imag -= tmp_c.imag;
	   	}
	    	
	    for(j = 0; j < n; j++)
	    {
	    	if(host_norm(tmp,n) > max)
	    		max = host_norm(tmp,n);
	    }
	}
	free(tmp);
	return max;
}

//run a single eigendecompositon on the GPU and check the result.
void single_eigenTest()
{
	#define N 8
	int i;
	Complex A[N*N], V[N*N], eigenValues[N];
	Complex *dev_A, *dev_V, *dev_Q, *dev_cv1, *dev_cv2, *dev_cv3, *dev_eigenVals;
	Real *dev_rv1;
	Real epsilon = EPS_M*2;

	//define my matrix on the CPU side
	INDEX(A,8,0,0).real = -1.2;
	INDEX(A,8,0,0).imag = 3.1;
	INDEX(A,8,0,1).real = 2.4;
	INDEX(A,8,0,1).imag = -0.1;
	INDEX(A,8,0,2).real = 0.3;
	INDEX(A,8,0,2).imag = -1.0;
	INDEX(A,8,0,3).real = 3.6;
	INDEX(A,8,0,3).imag = -0.2;
	INDEX(A,8,0,4).real = 1.0;
	INDEX(A,8,0,4).imag = 0.0;
	INDEX(A,8,0,5).real = 0.1;
	INDEX(A,8,0,5).imag = -3.2;
	INDEX(A,8,0,6).real = 2.3;
	INDEX(A,8,0,6).imag = 1.6;
	INDEX(A,8,0,7).real = 0.4;
	INDEX(A,8,0,7).imag = -2.3;
	INDEX(A,8,1,0).real = 1.1;
	INDEX(A,8,1,0).imag = -0.2;
	INDEX(A,8,1,1).real = -2.0;
	INDEX(A,8,1,1).imag = 1.0;
	INDEX(A,8,1,2).real = 3.2;
	INDEX(A,8,1,2).imag = 3.8;
	INDEX(A,8,1,3).real = -2.1;
	INDEX(A,8,1,3).imag = -1.2;
	INDEX(A,8,1,4).real = 1.6;
	INDEX(A,8,1,4).imag = 0.9;
	INDEX(A,8,1,5).real = -2.1;
	INDEX(A,8,1,5).imag = -0.1;
	INDEX(A,8,1,6).real = 1.3;
	INDEX(A,8,1,6).imag = 2.1;
	INDEX(A,8,1,7).real = 4.2;
	INDEX(A,8,1,7).imag = 0.0;

	INDEX(A,8,2,0).real = -1.2;
	INDEX(A,8,2,0).imag = 3.1;
	INDEX(A,8,2,1).real = 2.4;
	INDEX(A,8,2,1).imag = -0.1;
	INDEX(A,8,2,2).real = 0.3;
	INDEX(A,8,2,2).imag = -1.0;
	INDEX(A,8,2,3).real = 3.6;
	INDEX(A,8,2,3).imag = -0.2;
	INDEX(A,8,2,4).real = 1.0;
	INDEX(A,8,2,4).imag = 0.0;
	INDEX(A,8,2,5).real = 0.1;
	INDEX(A,8,2,5).imag = -3.2;
	INDEX(A,8,2,6).real = 0.0;
	INDEX(A,8,2,6).imag = 0.0;
	INDEX(A,8,2,7).real = 0.4;
	INDEX(A,8,2,7).imag = -2.3;
	INDEX(A,8,3,0).real = 1.1;
	INDEX(A,8,3,0).imag = -0.2;
	INDEX(A,8,3,1).real = -2.0;
	INDEX(A,8,3,1).imag = 1.0;
	INDEX(A,8,3,2).real = 3.2;
	INDEX(A,8,3,2).imag = 3.8;
	INDEX(A,8,3,3).real = -2.1;
	INDEX(A,8,3,3).imag = -1.2;
	INDEX(A,8,3,4).real = 1.6;
	INDEX(A,8,3,4).imag = 0.9;
	INDEX(A,8,3,5).real = -2.1;
	INDEX(A,8,3,5).imag = -0.1;
	INDEX(A,8,3,6).real = 1.3;
	INDEX(A,8,3,6).imag = 2.1;
	INDEX(A,8,3,7).real = 4.2;
	INDEX(A,8,3,7).imag = 0.0;

	INDEX(A,8,4,0).real = 1.2;
	INDEX(A,8,4,0).imag = 3.1;
	INDEX(A,8,4,1).real = 0.4;
	INDEX(A,8,4,1).imag = -0.1;
	INDEX(A,8,4,2).real = 0.3;
	INDEX(A,8,4,2).imag = -1.0;
	INDEX(A,8,4,3).real = 3.6;
	INDEX(A,8,4,3).imag = -0.2;
	INDEX(A,8,4,4).real = 0.0;
	INDEX(A,8,4,4).imag = 0.0;
	INDEX(A,8,4,5).real = 3.1;
	INDEX(A,8,4,5).imag = -3.2;
	INDEX(A,8,4,6).real = 2.3;
	INDEX(A,8,4,6).imag = 1.6;
	INDEX(A,8,4,7).real = 0.4;
	INDEX(A,8,4,7).imag = -2.3;
	INDEX(A,8,5,0).real = 2.1;
	INDEX(A,8,5,0).imag = -0.2;
	INDEX(A,8,5,1).real = -2.0;	//define my matrix on the CPU side
	INDEX(A,8,0,0).real = -1.2;
	INDEX(A,8,0,0).imag = 3.1;
	INDEX(A,8,0,1).real = 2.4;
	INDEX(A,8,0,1).imag = -0.1;
	INDEX(A,8,0,2).real = 0.3;
	INDEX(A,8,0,2).imag = -1.0;
	INDEX(A,8,0,3).real = 3.6;
	INDEX(A,8,0,3).imag = -0.2;
	INDEX(A,8,0,4).real = 1.0;
	INDEX(A,8,0,4).imag = 0.0;
	INDEX(A,8,0,5).real = 0.1;
	INDEX(A,8,0,5).imag = -3.2;
	INDEX(A,8,0,6).real = 2.3;
	INDEX(A,8,0,6).imag = 1.6;
	INDEX(A,8,0,7).real = 0.4;
	INDEX(A,8,0,7).imag = -2.3;
	INDEX(A,8,1,0).real = 1.1;
	INDEX(A,8,1,0).imag = -0.2;
	INDEX(A,8,1,1).real = -2.0;
	INDEX(A,8,1,1).imag = 1.0;
	INDEX(A,8,1,2).real = 3.2;
	INDEX(A,8,1,2).imag = 3.8;
	INDEX(A,8,1,3).real = -2.1;
	INDEX(A,8,1,3).imag = -1.2;
	INDEX(A,8,1,4).real = 1.6;
	INDEX(A,8,1,4).imag = 0.9;
	INDEX(A,8,1,5).real = -2.1;
	INDEX(A,8,1,5).imag = -0.1;
	INDEX(A,8,1,6).real = 1.3;
	INDEX(A,8,1,6).imag = 2.1;
	INDEX(A,8,1,7).real = 4.2;
	INDEX(A,8,1,7).imag = 0.0;

	INDEX(A,8,2,0).real = -1.2;
	INDEX(A,8,2,0).imag = 3.1;
	INDEX(A,8,2,1).real = 2.4;
	INDEX(A,8,2,1).imag = -0.1;
	INDEX(A,8,2,2).real = 0.3;
	INDEX(A,8,2,2).imag = -1.0;
	INDEX(A,8,2,3).real = 3.6;
	INDEX(A,8,2,3).imag = -0.2;
	INDEX(A,8,2,4).real = 1.0;
	INDEX(A,8,2,4).imag = 0.0;
	INDEX(A,8,2,5).real = 0.1;
	INDEX(A,8,2,5).imag = -3.2;
	INDEX(A,8,2,6).real = 0.0;
	INDEX(A,8,2,6).imag = 0.0;
	INDEX(A,8,2,7).real = 0.4;
	INDEX(A,8,2,7).imag = -2.3;
	INDEX(A,8,3,0).real = 1.1;
	INDEX(A,8,3,0).imag = -0.2;
	INDEX(A,8,3,1).real = -2.0;
	INDEX(A,8,3,1).imag = 1.0;
	INDEX(A,8,3,2).real = 3.2;
	INDEX(A,8,3,2).imag = 3.8;
	INDEX(A,8,3,3).real = -2.1;
	INDEX(A,8,3,3).imag = -1.2;
	INDEX(A,8,3,4).real = 1.6;
	INDEX(A,8,3,4).imag = 0.9;
	INDEX(A,8,3,5).real = -2.1;
	INDEX(A,8,3,5).imag = -0.1;
	INDEX(A,8,3,6).real = 1.3;
	INDEX(A,8,3,6).imag = 2.1;
	INDEX(A,8,3,7).real = 4.2;
	INDEX(A,8,3,7).imag = 0.0;

	INDEX(A,8,4,0).real = 1.2;
	INDEX(A,8,4,0).imag = 3.1;
	INDEX(A,8,4,1).real = 0.4;
	INDEX(A,8,4,1).imag = -0.1;
	INDEX(A,8,4,2).real = 0.3;
	INDEX(A,8,4,2).imag = -1.0;
	INDEX(A,8,4,3).real = 3.6;
	INDEX(A,8,4,3).imag = -0.2;
	INDEX(A,8,4,4).real = 0.0;
	INDEX(A,8,4,4).imag = 0.0;
	INDEX(A,8,4,5).real = 3.1;
	INDEX(A,8,4,5).imag = -3.2;
	INDEX(A,8,4,6).real = 2.3;
	INDEX(A,8,4,6).imag = 1.6;
	INDEX(A,8,4,7).real = 0.4;
	INDEX(A,8,4,7).imag = -2.3;
	INDEX(A,8,5,0).real = 2.1;
	INDEX(A,8,5,0).imag = -0.2;
	INDEX(A,8,5,1).real = -2.0;
	INDEX(A,8,5,1).imag = 1.0;
	INDEX(A,8,5,2).real = 3.2;
	INDEX(A,8,5,2).imag = -3.8;
	INDEX(A,8,5,3).real = 2.1;
	INDEX(A,8,5,3).imag = -1.2;
	INDEX(A,8,5,4).real = 1.6;
	INDEX(A,8,5,4).imag = 0.9;
	INDEX(A,8,5,5).real = 0.0;
	INDEX(A,8,5,5).imag = 0.0;
	INDEX(A,8,5,6).real = 1.3;
	INDEX(A,8,5,6).imag = 2.1;
	INDEX(A,8,5,7).real = -4.2;
	INDEX(A,8,5,7).imag = 0.0;

	INDEX(A,8,6,0).real = -1.2;
	INDEX(A,8,6,0).imag = 3.1;
	INDEX(A,8,6,1).real = -2.4;
	INDEX(A,8,6,1).imag = -0.1;
	INDEX(A,8,6,2).real = 0.3;
	INDEX(A,8,6,2).imag = -1.6;
	INDEX(A,8,6,3).real = 0.0;
	INDEX(A,8,6,3).imag = 0.0;
	INDEX(A,8,6,4).real = 1.0;
	INDEX(A,8,6,4).imag = 0.7;
	INDEX(A,8,6,5).real = 0.1;
	INDEX(A,8,6,5).imag = -3.2;
	INDEX(A,8,6,6).real = 2.3;
	INDEX(A,8,6,6).imag = -1.6;
	INDEX(A,8,6,7).real = -0.4;
	INDEX(A,8,6,7).imag = 2.3;
	INDEX(A,8,7,0).real = 1.1;
	INDEX(A,8,7,0).imag = 0.2;
	INDEX(A,8,7,1).real = 0.0;
	INDEX(A,8,7,1).imag = 0.0;
	INDEX(A,8,7,2).real = -3.2;
	INDEX(A,8,7,2).imag = -3.8;
	INDEX(A,8,7,3).real = -2.1;
	INDEX(A,8,7,3).imag = -1.2;
	INDEX(A,8,7,4).real = 1.6;
	INDEX(A,8,7,4).imag = 2.9;
	INDEX(A,8,7,5).real = -2.1;
	INDEX(A,8,7,5).imag = -0.1;
	INDEX(A,8,7,6).real = 1.3;
	INDEX(A,8,7,6).imag = 0.1;
	INDEX(A,8,7,7).real = 4.2;
	INDEX(A,8,7,7).imag = 0.0;
	INDEX(A,8,5,1).imag = 1.0;
	INDEX(A,8,5,2).real = 3.2;
	INDEX(A,8,5,2).imag = -3.8;
	INDEX(A,8,5,3).real = 2.1;
	INDEX(A,8,5,3).imag = -1.2;
	INDEX(A,8,5,4).real = 1.6;
	INDEX(A,8,5,4).imag = 0.9;
	INDEX(A,8,5,5).real = 0.0;
	INDEX(A,8,5,5).imag = 0.0;
	INDEX(A,8,5,6).real = 1.3;
	INDEX(A,8,5,6).imag = 2.1;
	INDEX(A,8,5,7).real = -4.2;
	INDEX(A,8,5,7).imag = 0.0;

	INDEX(A,8,6,0).real = -1.2;
	INDEX(A,8,6,0).imag = 3.1;
	INDEX(A,8,6,1).real = -2.4;
	INDEX(A,8,6,1).imag = -0.1;
	INDEX(A,8,6,2).real = 0.3;
	INDEX(A,8,6,2).imag = -1.6;
	INDEX(A,8,6,3).real = 0.0;
	INDEX(A,8,6,3).imag = 0.0;
	INDEX(A,8,6,4).real = 1.0;
	INDEX(A,8,6,4).imag = 0.7;
	INDEX(A,8,6,5).real = 0.1;
	INDEX(A,8,6,5).imag = -3.2;
	INDEX(A,8,6,6).real = 2.3;
	INDEX(A,8,6,6).imag = -1.6;
	INDEX(A,8,6,7).real = -0.4;
	INDEX(A,8,6,7).imag = 2.3;
	INDEX(A,8,7,0).real = 1.1;
	INDEX(A,8,7,0).imag = 0.2;
	INDEX(A,8,7,1).real = 0.0;
	INDEX(A,8,7,1).imag = 0.0;
	INDEX(A,8,7,2).real = -3.2;
	INDEX(A,8,7,2).imag = -3.8;
	INDEX(A,8,7,3).real = -2.1;
	INDEX(A,8,7,3).imag = -1.2;
	INDEX(A,8,7,4).real = 1.6;
	INDEX(A,8,7,4).imag = 2.9;
	INDEX(A,8,7,5).real = -2.1;
	INDEX(A,8,7,5).imag = -0.1;
	INDEX(A,8,7,6).real = 1.3;
	INDEX(A,8,7,6).imag = 0.1;
	INDEX(A,8,7,7).real = 4.2;
	INDEX(A,8,7,7).imag = 0.0;

	//Allocate memory on the GPU
	HANDLE_ERROR( cudaMalloc( (void**)&dev_A, N * N * sizeof(Complex) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_V, N * N * sizeof(Complex) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_Q, N * N * sizeof(Complex) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_cv1, N * sizeof(Complex) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_cv2, N * sizeof(Complex) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_cv3, N * sizeof(Complex) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_eigenVals, N * sizeof(Complex) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_rv1, N * sizeof(Real) ) );

	//Copy A over to the GPU
	HANDLE_ERROR( cudaMemcpy( dev_A, A, N*N * sizeof(Complex), cudaMemcpyHostToDevice ) );

	//Run one instance of the algorithm on the GPU
	eigen_caller<<<1,1>>>(dev_A, dev_V, dev_Q, dev_eigenVals, dev_rv1, dev_cv1, dev_cv2, dev_cv3, N, epsilon, 50);

	//Copy the result back to the host computer
    HANDLE_ERROR( cudaMemcpy( V, dev_V, N*N * sizeof(Complex), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( eigenValues, dev_eigenVals, N * sizeof(Complex), cudaMemcpyDeviceToHost ) );

	//Check the result!
	printf("Original Matrix:\n");
	printMat(A,N);

	printf("Eigenvalues:\n");
	for(i = 0; i < N; i++)
	{
		printf("%1.4f + %1.4fi\n", eigenValues[i].real, eigenValues[i].imag);
	}
	
	printf("Eigenvectors:\n");
	printMat(V,N);

	printf("max norm for Av-lv: %f\n",eigenTest(A,V,eigenValues,N));
}

//----- Timing ----

//taken from:
//http://stackoverflow.com/questions/1468596/c-programming-calculate-elapsed-time-in-milliseconds-unix
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
    result->tv_sec = diff / 1000000;
    result->tv_usec = diff % 1000000;

    return (diff<0);
}

#include <sys/time.h>
//run a whole bunch (num) of eigendecompositons on the GPU at the same time and
//return the time it took.
double multi_eigenTest(int num)
{
	//linux wall time
	struct timeval tvBegin, tvEnd, tvDiff;

	#define N 8
	int i,j;
	Complex A[N*N], V[N*N], eigenValues[N];
	Complex *dev_A, *dev_V, *dev_Q, *dev_cv1, *dev_cv2, *dev_cv3, *dev_eigenVals;
	Real *dev_rv1;
	Real epsilon = EPS_M*2;

	//define my matrix on the CPU side
	INDEX(A,8,0,0).real = -1.2;
	INDEX(A,8,0,0).imag = 3.1;
	INDEX(A,8,0,1).real = 2.4;
	INDEX(A,8,0,1).imag = -0.1;
	INDEX(A,8,0,2).real = 0.3;
	INDEX(A,8,0,2).imag = -1.0;
	INDEX(A,8,0,3).real = 3.6;
	INDEX(A,8,0,3).imag = -0.2;
	INDEX(A,8,0,4).real = 1.0;
	INDEX(A,8,0,4).imag = 0.0;
	INDEX(A,8,0,5).real = 0.1;
	INDEX(A,8,0,5).imag = -3.2;
	INDEX(A,8,0,6).real = 2.3;
	INDEX(A,8,0,6).imag = 1.6;
	INDEX(A,8,0,7).real = 0.4;
	INDEX(A,8,0,7).imag = -2.3;
	INDEX(A,8,1,0).real = 1.1;
	INDEX(A,8,1,0).imag = -0.2;
	INDEX(A,8,1,1).real = -2.0;
	INDEX(A,8,1,1).imag = 1.0;
	INDEX(A,8,1,2).real = 3.2;
	INDEX(A,8,1,2).imag = 3.8;
	INDEX(A,8,1,3).real = -2.1;
	INDEX(A,8,1,3).imag = -1.2;
	INDEX(A,8,1,4).real = 1.6;
	INDEX(A,8,1,4).imag = 0.9;
	INDEX(A,8,1,5).real = -2.1;
	INDEX(A,8,1,5).imag = -0.1;
	INDEX(A,8,1,6).real = 1.3;
	INDEX(A,8,1,6).imag = 2.1;
	INDEX(A,8,1,7).real = 4.2;
	INDEX(A,8,1,7).imag = 0.0;

	INDEX(A,8,2,0).real = -1.2;
	INDEX(A,8,2,0).imag = 3.1;
	INDEX(A,8,2,1).real = 2.4;
	INDEX(A,8,2,1).imag = -0.1;
	INDEX(A,8,2,2).real = 0.3;
	INDEX(A,8,2,2).imag = -1.0;
	INDEX(A,8,2,3).real = 3.6;
	INDEX(A,8,2,3).imag = -0.2;
	INDEX(A,8,2,4).real = 1.0;
	INDEX(A,8,2,4).imag = 0.0;
	INDEX(A,8,2,5).real = 0.1;
	INDEX(A,8,2,5).imag = -3.2;
	INDEX(A,8,2,6).real = 0.0;
	INDEX(A,8,2,6).imag = 0.0;
	INDEX(A,8,2,7).real = 0.4;
	INDEX(A,8,2,7).imag = -2.3;
	INDEX(A,8,3,0).real = 1.1;
	INDEX(A,8,3,0).imag = -0.2;
	INDEX(A,8,3,1).real = -2.0;
	INDEX(A,8,3,1).imag = 1.0;
	INDEX(A,8,3,2).real = 3.2;
	INDEX(A,8,3,2).imag = 3.8;
	INDEX(A,8,3,3).real = -2.1;
	INDEX(A,8,3,3).imag = -1.2;
	INDEX(A,8,3,4).real = 1.6;
	INDEX(A,8,3,4).imag = 0.9;
	INDEX(A,8,3,5).real = -2.1;
	INDEX(A,8,3,5).imag = -0.1;
	INDEX(A,8,3,6).real = 1.3;
	INDEX(A,8,3,6).imag = 2.1;
	INDEX(A,8,3,7).real = 4.2;
	INDEX(A,8,3,7).imag = 0.0;

	INDEX(A,8,4,0).real = 1.2;
	INDEX(A,8,4,0).imag = 3.1;
	INDEX(A,8,4,1).real = 0.4;
	INDEX(A,8,4,1).imag = -0.1;
	INDEX(A,8,4,2).real = 0.3;
	INDEX(A,8,4,2).imag = -1.0;
	INDEX(A,8,4,3).real = 3.6;
	INDEX(A,8,4,3).imag = -0.2;
	INDEX(A,8,4,4).real = 0.0;
	INDEX(A,8,4,4).imag = 0.0;
	INDEX(A,8,4,5).real = 3.1;
	INDEX(A,8,4,5).imag = -3.2;
	INDEX(A,8,4,6).real = 2.3;
	INDEX(A,8,4,6).imag = 1.6;
	INDEX(A,8,4,7).real = 0.4;
	INDEX(A,8,4,7).imag = -2.3;
	INDEX(A,8,5,0).real = 2.1;
	INDEX(A,8,5,0).imag = -0.2;
	INDEX(A,8,5,1).real = -2.0;
	INDEX(A,8,5,1).imag = 1.0;
	INDEX(A,8,5,2).real = 3.2;
	INDEX(A,8,5,2).imag = -3.8;
	INDEX(A,8,5,3).real = 2.1;
	INDEX(A,8,5,3).imag = -1.2;
	INDEX(A,8,5,4).real = 1.6;
	INDEX(A,8,5,4).imag = 0.9;
	INDEX(A,8,5,5).real = 0.0;
	INDEX(A,8,5,5).imag = 0.0;
	INDEX(A,8,5,6).real = 1.3;
	INDEX(A,8,5,6).imag = 2.1;
	INDEX(A,8,5,7).real = -4.2;
	INDEX(A,8,5,7).imag = 0.0;

	INDEX(A,8,6,0).real = -1.2;
	INDEX(A,8,6,0).imag = 3.1;
	INDEX(A,8,6,1).real = -2.4;
	INDEX(A,8,6,1).imag = -0.1;
	INDEX(A,8,6,2).real = 0.3;
	INDEX(A,8,6,2).imag = -1.6;
	INDEX(A,8,6,3).real = 0.0;
	INDEX(A,8,6,3).imag = 0.0;
	INDEX(A,8,6,4).real = 1.0;
	INDEX(A,8,6,4).imag = 0.7;
	INDEX(A,8,6,5).real = 0.1;
	INDEX(A,8,6,5).imag = -3.2;
	INDEX(A,8,6,6).real = 2.3;
	INDEX(A,8,6,6).imag = -1.6;
	INDEX(A,8,6,7).real = -0.4;
	INDEX(A,8,6,7).imag = 2.3;
	INDEX(A,8,7,0).real = 1.1;
	INDEX(A,8,7,0).imag = 0.2;
	INDEX(A,8,7,1).real = 0.0;
	INDEX(A,8,7,1).imag = 0.0;
	INDEX(A,8,7,2).real = -3.2;
	INDEX(A,8,7,2).imag = -3.8;
	INDEX(A,8,7,3).real = -2.1;
	INDEX(A,8,7,3).imag = -1.2;
	INDEX(A,8,7,4).real = 1.6;
	INDEX(A,8,7,4).imag = 2.9;
	INDEX(A,8,7,5).real = -2.1;
	INDEX(A,8,7,5).imag = -0.1;
	INDEX(A,8,7,6).real = 1.3;
	INDEX(A,8,7,6).imag = 0.1;
	INDEX(A,8,7,7).real = 4.2;
	INDEX(A,8,7,7).imag = 0.0;


//	gettimeofday(&tvBegin, NULL);//linux wall time

	//Allocate memory on the GPU
	HANDLE_ERROR( cudaMalloc( (void**)&dev_A, num * N * N * sizeof(Complex) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_V, num * N * N * sizeof(Complex) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_Q, num * N * N * sizeof(Complex) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_cv1, num * N * sizeof(Complex) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_cv2, num * N * sizeof(Complex) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_cv3, num * N * sizeof(Complex) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_eigenVals, num * N * sizeof(Complex) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_rv1, num * N * sizeof(Real) ) );

	//Copy A over to the GPU
	for(i = 0; i < num; i++)
		HANDLE_ERROR( cudaMemcpy( &dev_A[i*N*N], A, N*N * sizeof(Complex), cudaMemcpyHostToDevice ) );

	gettimeofday(&tvBegin, NULL);//linux wall time	

	//Run one instance of the algorithm on the GPU
	
	eigen_caller<<<num/10,10>>>(dev_A, dev_V, dev_Q, dev_eigenVals, dev_rv1, dev_cv1, dev_cv2, dev_cv3, N, epsilon, 50);
	HANDLE_ERROR( cudaThreadSynchronize() );

	gettimeofday(&tvEnd, NULL);//linux wall time
	// diff
    timeval_subtract(&tvDiff, &tvEnd, &tvBegin);

	//copy back to host and test
	for(j = 0; j < num; j++)
	{
		//Copy the result back to the host computer
		HANDLE_ERROR( cudaMemcpy( V, &dev_V[j*N*N], N*N * sizeof(Complex), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy( eigenValues, &dev_eigenVals[j*N], N * sizeof(Complex), cudaMemcpyDeviceToHost ) );

		
		if(eigenTest(A,V,eigenValues,N) > 0.001)//lazy test
		{
			printf("AAHHHHHHH %f!!!\n", eigenTest(A,V,eigenValues,N));
//			printf("A:\n");
//			printMat(A, N);
//			printf("V:\n");
//			printMat(V, N);

		}
	}

	HANDLE_ERROR( cudaFree( dev_A ) );
	HANDLE_ERROR( cudaFree( dev_V ) );
	HANDLE_ERROR( cudaFree( dev_Q ) );
	HANDLE_ERROR( cudaFree( dev_cv1 ) );
	HANDLE_ERROR( cudaFree( dev_cv2 ) );
	HANDLE_ERROR( cudaFree( dev_cv3 ) );
	HANDLE_ERROR( cudaFree( dev_eigenVals ) );
	HANDLE_ERROR( cudaFree( dev_rv1 ) );



	//return elapsedTime;
	return tvDiff.tv_sec*1000 + tvDiff.tv_usec/1000;
}


__host__ void timeProfile()
{
	int i;
	double val;
	FILE *file; 
	file = fopen("my_custom_zgeev_times.txt","w"); /* apend file (add text to 

	a file or create a file if it does not exist.*/ 

	for(i = 100000; i <= 200000; i+=100000)
	{
		val = multi_eigenTest(i);
		printf("%f milliseconds to do %d\n",val, i);
		fprintf(file,"%d\t%f\n",i,val); /*writes*/ 
	}
	fclose(file); /*done!*/ 
}
/*
int main()
{

//	single_eigenTest();

	srand(time(NULL));
	timeProfile();

	return 0;
}*/
