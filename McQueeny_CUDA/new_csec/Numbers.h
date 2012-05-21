#ifndef NUMBERS_H
#define NUMBERS_H

//32 bit precision has been giving me problems - sometimes the eigendecomp doesn't converge
//typedef float Real;
typedef double Real;  //compile with "nvcc -arch sm_13" to use doubles

typedef struct
{
	Real real;
	Real imag;
}Complex;

//defined by G.W. Stewart as a small number safely above
//the underflow point (I don't know exactly what "safely" means)
//technically this should depend on the hardware/C compiler
//#define SMALLNUM 1e-35
//#define BIGNUM 1e35
//#define EPS_M 1e-7//single precision float has 7.225 decimal digits precision
//Note:  (4/23/12) As of right now, double precision does not work.  I think
//nvcc is automatically converting a double somewhere to a float (it says it is)
//and then the precision values below are too small, so I just get junk out.
//It may be possible to change this with a compiler option - I think old
//cards cannot use doubles, but new ones can and an nvcc flag sets this
#define SMALLNUM 1e-300
#define BIGNUM 1e300//"a number near the overflow point"
#define EPS_M 1e-14//This is probably not optimal

#define ABS(A) sqrt(A.real*A.real + A.imag*A.imag)



//Helper Functions

__device__ Complex complexMult(Complex a, Complex b)
{
	Complex result;
	result.real = a.real*b.real - a.imag*b.imag;
	result.imag = a.imag*b.real + a.real*b.imag;
	return result;
}


#endif /* NUMBERS_H */
