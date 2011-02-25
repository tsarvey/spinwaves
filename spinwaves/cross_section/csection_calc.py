"""

Disclaimer
==========

This software was developed at the National Institute of Standards and Technology at the NIST Center for 
Neutron Research by employees of the Federal Government in the course of their official duties. Pursuant 
to title 17 section 105* of the United States Code this software is not subject to copyright protection 
and is in the public domain. The SPINAL software package is an experimental spinwave analysis system. NIST 
assumes no responsibility whatsoever for its use, and makes no guarantees, expressed or implied, about its 
quality, reliability, or any other characteristic. The use of certain trade names or commercial products 
does not imply any endorsement of a particular product, nor does it imply that the named product is 
necessarily the best product for the stated purpose. We would appreciate acknowledgment if the software is 
used.

*Subject matter of copyright: United States Government works

Copyright protection under this title is not available for any work of the United States Government, but the 
United States Government is not precluded from receiving and holding copyrights transferred to it by 
assignment, bequest, or otherwise

Author: wflynn
"""

import sys
import os
import time
#sys.path.append(r"C:\Documents and Settings\wflynn\My Documents\workspace\spinwaves_SVN")
sys.path.append(r"/home/tom/Documents/code/checkout/")

import subprocess
import sympy as sp
import sympy.matrices as spm
from sympy import I,pi,exp,oo,sqrt,abs,S,Pow,re,Symbol,Wild
from sympy.physics.paulialgebra import delta
from sympy.core.cache import clear_cache
import numpy as np
from numpy import arctan2,sin,cos,array
from numpy import newaxis as nax
from scipy.integrate import simps, trapz, fixed_quad

import matplotlib
matplotlib.use('WXAgg')
import pylab
from matplotlib._pylab_helpers import Gcf
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib import pyplot
testing = True
if testing:
    import sys
    sys.path.append("C:\\Documents and Settings\\tsarvey\\My Documents\\spinwaves")

from spinwaves.cross_section.util.subin import sub_in
from spinwaves.spinwavecalc.readfiles import atom, readFiles
from spinwaves.spinwavecalc.spinwave_calc_file import calculate_dispersion, calc_eigs_direct
from rescalculator.lattice_calculator import Lattice, Orientation
from periodictable import elements

from multiprocessing import Pipe, Process
from copy import deepcopy,copy
from timeit import default_timer as clock

from mcQueeny_alg import mcQueeny_case

#from cluster.william_mapper import Mapper
#
#import Pyro.core
#import Pyro.errors
#Pyro.core.initClient()
#
#import psyco
#psyco.log()
#psyco.full(memory=1000000)
#psyco.profile(0.05, memory=100)
#psyco.profile(0.2)

#------------ GLOBAL VARIABLES ---------------------------------------
"""
 ALL GLOBACL VARIABLES ARE CAPTIALIZED
 ALL GLOBAL SYMBOLS WILL BE IN CAPS, FOLLOWED BY _SYM.
 ALL OTHER SYMBOLS ARE USUALLY CONSTRUCTED WITH 'sym%i'%(num,)
"""

Testing = True

# define global variables
T_SYM = Symbol('t', real = True) # t
L_SYM = Symbol('L', real = True) # L
Q_SYM = Symbol('q', real = True) # q
QP_SYM = Symbol('qp', real = True) # qp
W_SYM = Symbol('w', real = True)
WQ_SYM = Symbol('wq', real = True) # wq
WQP_SYM = Symbol('wqp', real = True) #wqp
WT_SYM = Symbol('wt', real = True)
S_SYM = Symbol('S', commutative = True) # S
NQ_SYM = Symbol('nq', real = True) #nq = n0, n1, n2, ...
TEMP_SYM = Symbol('T', real = True)

KAP_SYM = Symbol('kappa', real = True) # kap
TAU_SYM = Symbol('tau', real = True) # tau
KX_SYM = Symbol('kx', real = True)
KY_SYM = Symbol('ky', real = True)
KZ_SYM = Symbol('kz', real = True)
DD_KTPQ_SYM = Symbol('DDKTPQ', real = True)
DD_KTMQ_SYM = Symbol('DDKTMQ', real = True)

THETA_SYM = sp.Symbol('theta', real = True)
PHI_SYM = sp.Symbol('phi', real = True)
EIG_SYM = sp.Symbol('eig', real = True)
RAD_SYM = sp.Symbol('rad', real = True)
FF_SYM = sp.Symbol('ff', real = True)

KAPXHAT_SYM = Symbol('kapxhat',real=True)
KAPYHAT_SYM = Symbol('kapyhat',real=True)
KAPZHAT_SYM = Symbol('kapzhat',real=True)

#line 821
#QR_DOT_SYM = Symbol('Q.R', real = True)#This is the Q vecter dotted with the basis vector
R_SYM = Symbol('R', real = True)#Tom 2/11/11

# Wilds for sub_in method
A_WILD = Wild('A',exclude = [0,T_SYM])
B_WILD = Wild('B',exclude = [0,T_SYM])
C_WILD = Wild('C')
D_WILD = Wild('D')
K_WILD = Wild('K')
    
GAMMA_R0_VALUE = 1.913*2.818
HBAR_VALUE = 1.0#6.582e-13 
G_VALUE = 2.
LIFETIME_VALUE = 0.5
BOLTZ_VALUE = 8.617343e-2
DEBYE_WALLER_VALUE = 1.0 #exp(-2*W)

#------------ HELPER METHODS ---------------------------------------------------

def list_print(lista):
    print 'printing...'
    for element in lista:
        print element
    print ''

def list_mult(lista, listb):
    "Defines a way to multiply two lists of the same length"
    if len(lista) != len(listb):
        print "lists not same length"
        return []
    else:
        temp = []
        for i in range(len(lista)):
            if isinstance(lista[i], int):
                temp.append(sp.powsimp((lista[i] * listb[i]).expand(deep=False)))
            elif isinstance(listb[i], int):
                temp.append(sp.powsimp((lista[i] * listb[i]).expand(deep=False)))
            else:
                temp.append(sp.powsimp((lista[i] * listb[i]).expand(deep=False)))
        return temp

def coeff(expr, term):
    "Returns the coefficient of a term given an expression"
    if isinstance(expr, int):
        return 0
    expr = sp.collect(expr, term)
    symbols = list(term.atoms(sp.Symbol))
    w = Wild("coeff", exclude = symbols)
    m = expr.match(w * term + sp.Wild("rest"))
    m2 = expr.match(w * term)
    res = False
    if m2:
        res = m2[w] * term == expr
    if m and not res:
        return m[w]
    #added the next two lines
    elif m2:
        return m2[w]
    
def coeff_bins(expr,bins):
    # chop up expr at '+'
    expr_list = sp.make_list(expr, sp.Add)
    retbins = np.zeros(len(bins),dtype=object)
    # get rid of None expressions
    if expr is None or isinstance(expr,int):
        return retbins
    #scan through expr
    for subexpr in expr_list:
        #see if it contains a bin element
        for i in range(len(bins)):
            curr_coeff = subexpr.as_coefficient(bins[i])
            if curr_coeff:
                retbins[i] += curr_coeff
    return retbins

def chop(expr, eps):
    cls = expr.__class__
    newexpr = 0
    terms = expr.args
    for term in terms:
        if term.as_coeff_terms()[0] >= eps:
            newexpr += term
    return newexpr

def generate_lattice():
    aa = bb = cc = np.array([2.0*np.pi], 'Float64')
    alpha = beta = gamma = np.array([np.pi/2.0], 'Float64')
    vect1 = np.array([[1,0,0]])
    vect2 = np.array([[0,0,1]])
    lattice = Lattice(aa, bb, cc, alpha, beta, gamma, Orientation(vect1, vect2))
    return lattice


#------------ CROSS SECTION CALC METHODS ---------------------------------------

# Lists of the b and b dagger operators
def generate_b_bd_operators(atom_list):
    """Generates b and b dagger operators"""
    b_list = []; bd_list = []
    N = len(atom_list)
    for i in range(N):
        b = Symbol('b%i'%(i,), commutative = False)
        bd = Symbol('bd%i'%(i,), commutative = False)
        b_list.append(b); bd_list.append(bd)

    print "Operators Generated: b, bd"
    return (b_list,bd_list)

# Generates the a and a dagger operators
def generate_a_ad_operators(atom_list, k, b_list, bd_list):
    """Generates a and a dagger operators"""
    a_list = []; ad_list = []
    a0_list = []; ad0_list = []
    N = len(atom_list)

    for i in range(N):
        a_list.append(exp(I*(QP_SYM*L_SYM - WQP_SYM*T_SYM)) * b_list[i])
        ad_list.append(exp(-I*(QP_SYM*L_SYM - WQP_SYM*T_SYM)) * bd_list[i])
        a0_list.append(exp(0) * b_list[i])
        ad0_list.append(exp(0) * bd_list[i])

    a = Pow(sp.sqrt(N),-1) * sum(a_list)
    ad = Pow(sp.sqrt(N),-1) * sum(ad_list)
    a0 = Pow(sp.sqrt(N),-1) * sum(a0_list)
    ad0 = Pow(sp.sqrt(N),-1) * sum(ad0_list)

    print "Operators Generated: a, ad"
    return (a, ad, a0, ad0)

# Generates the Sp and Sm operators
def generate_Sp_Sm_operators(atom_list, a, ad, a0, ad0):
    """Generates S+ and S- operators"""

    Sp = sqrt(2*S_SYM) * a
    Sm = sqrt(2*S_SYM) * ad
    Sp0 = sqrt(2*S_SYM) * a0
    Sm0 = sqrt(2*S_SYM) * ad0

    print "Operators Generated: Sp, Sm"
    return (Sp, Sm, Sp0, Sm0)

def generate_Sa_Sb_Sn_operators(atom_list, Sp, Sm, Sp0, Sm0):
    """Generates Sa, Sb, Sn operators"""

    Sa = ((1./2.)*(Sp+Sm))#.expand()
    Sb = ((1./2.)*(1./I)*(Sp-Sm))#.expand()
    Sn = (S_SYM - Pow(2*S_SYM,-1) * Sm * Sp)#.expand()
    Sa0 = ((1./2.)*(Sp0+Sm0))#.expand()
    Sb0 = ((1./2.)*(1./I)*(Sp0-Sm0))#.expand()
    Sn0 = (S_SYM - Pow(2*S_SYM,-1) * Sm0 * Sp0)#.expand()
    
    print "Operators Generated: Sa, Sb, Sn"
    return (Sa, Sb, Sn, Sa0, Sb0, Sn0)

# Generates the Sx, Sy and Sz operators
def generate_Sx_Sy_Sz_operators(atom_list, Sa, Sb, Sn, Sa0, Sb0, Sn0):#, eps=0.0):
    """Generates Sx, Sy and Sz operators"""
    Sx_list = []; Sy_list = []; Sz_list = []
    Sx0_list = []; Sy0_list = []; Sz0_list = []
    N = len(atom_list)

    loc_vect = spm.Matrix([Sa,Sb,Sn])
    loc_vect = loc_vect.reshape(3,1)
    loc_vect0 = spm.Matrix([Sa0,Sb0,Sn0])
    loc_vect0 = loc_vect0.reshape(3,1)

    for i in range(N):
        rotmat = sp.Matrix(atom_list[i].spinRmatrix)
        glo_vect = rotmat * loc_vect
        glo_vect0 = rotmat * loc_vect0

#        Sx = sp.powsimp(glo_vect[0].expand())
#        Sy = sp.powsimp(glo_vect[1].expand())
#        Sz = sp.powsimp(glo_vect[2].expand())
        Sx = glo_vect[0].expand(mul=True,deep=False,power_exp=False)
        Sy = glo_vect[1].expand(mul=True,deep=False,power_exp=False)
        Sz = glo_vect[2].expand(mul=True,deep=False,power_exp=False)
#        Sx_list.append(chop(Sx,eps))
#        Sy_list.append(chop(Sy,eps))
#        Sz_list.append(chop(Sz,eps))
        Sx_list.append(Sx)
        Sy_list.append(Sy)
        Sz_list.append(Sz)

#        Sx0 = sp.powsimp(glo_vect0[0].expand())
#        Sy0 = sp.powsimp(glo_vect0[1].expand())
#        Sz0 = sp.powsimp(glo_vect0[2].expand())
        Sx0 = glo_vect0[0].expand(mul=True,deep=False,power_exp=False)
        Sy0 = glo_vect0[1].expand(mul=True,deep=False,power_exp=False)
        Sz0 = glo_vect0[2].expand(mul=True,deep=False,power_exp=False)
#        Sx0_list.append(chop(Sx0,eps))
#        Sy0_list.append(chop(Sy0,eps))
#        Sz0_list.append(chop(Sz0,eps))
        Sx0_list.append(Sx0)
        Sy0_list.append(Sy0)
        Sz0_list.append(Sz0)
          
    Sx_list.append(KAPXHAT_SYM)
    Sy_list.append(KAPYHAT_SYM)
    Sz_list.append(KAPZHAT_SYM)
    Sx0_list.append(KAPXHAT_SYM)
    Sy0_list.append(KAPYHAT_SYM)
    Sz0_list.append(KAPZHAT_SYM)
    
    print "Operators Generated: Sx, Sy, Sz"
    return (Sx_list,Sy_list,Sz_list,Sx0_list,Sy0_list,Sz0_list)

# Define a method that generates the possible combinations of operators
#def generate_possible_combinations(atom_list, alist):
def generate_possible_combinations(atom_list, op_list, op_list0):
    """This method returns the possible operator combinations from a list of operators"""
    # For a combination to be returned, the product must have an equal number of b
    # and b dagger operators. If not, they are rejected.
    res_list = []
    N = len(atom_list)
    
    op_list = np.array(op_list)
    op_list0 = np.array(op_list0)

    for i in range(len(op_list0)):
        vecti = op_list0[i,-1]
        for j in range(len(op_list)):
            vectj = op_list[j,-1]
            if cmp(vecti,vectj) == 0: delta = 1
            else: delta = 0

            res_list.append((op_list0[i,:-1]*op_list[j,:-1]).tolist() + [delta - vecti*vectj])
    
    res_list = map(sp.expand, res_list)
    
    print "Generated: Possible Operator Combinations"
    return res_list
 
def holstein(atom_list, arg):
    N = len(atom_list)
    arg = np.array(arg)

    for k in range(len(arg)):
        for i in range(N):
            Snew = atom_list[i].spinMagnitude
            
            #gets rid of time independent terms
            #works since arg[k][i] is an Add instance so args breaks it up at '+'s
            pieces = arg[k][i].args
            for piece in pieces:
                if not piece.has(T_SYM):
                    arg[k][i] = arg[k][i] - piece             

            coeffs = coeff_bins(arg[k][i],[S_SYM**2,S_SYM])
            S2coeff,Scoeff = coeffs[0],coeffs[1]
            if S2coeff and Scoeff:
                arg[k][i] = (S2coeff*Snew**2 + Scoeff*Snew)
            elif S2coeff and not Scoeff:
                arg[k][i] = (S2coeff*Snew**2)
            elif not S2coeff and Scoeff:
                arg[k][i] = (Scoeff*Snew)
            else:
                arg[k][i] = sp.S(0)
            # This does not account for None's
#            val = S2coeff*Snew**2 + Scoeff*Snew
#            if not val: val = sp.S(0)
#            arg[k][i] = val

    #removes all rows with zeros for each element
#    arg = arg[arg[:,:-1].any(axis=1)]
    
    print "Applied: Holstein"
    return arg.tolist()

def reduce_options(atom_list, arg):
    """
    Further reduces possible operator combinations by removing combinations if
    they are the negative of another combination or they are not time dependent
    (i.e. elastic scattering)
    """
#    new = []
#    N = len(atom_list)
#    for element in arg:
#        if str(element[0]).find('t') > 0:
#            new.append(element)
    new = arg

    for elementa in new:
        if elementa[0] == 0:
            new.remove(elementa)
            break
        # useful idea but implementation breaks. i also am not sure it saves too much time
#        for elementb in new:
#            if elementa[0].expand(deep = False) == (-1*elementb[0]).expand(deep = False):
#                print elementa[0], elementa[-1]
#                print elementb[0], elementb[-1]
#                new.remove(elementa)
#                new.remove(elementb)
#                break
    print 'Applied: Possible Operator Reduction'
    return new

# Apply Commutation Relation
def apply_commutation(atom_list, arg):
    """Applies the commutation relation of [b_i, bd_j] = kronecker delta _ ij"""
    # [bi,bdj] = delta_ij
    # Thus commutator = 0 (THEY COMMUTE) for i != j
    # Thus commutator = 1 for i == j
        # Then just put '+1' after commutation
    # NOTE: This method will take bd*b*bd*b to bd*(bd*b+1)*b so
    # I have replace bd_b called first but implement it inside this method too.
    N = len(atom_list)
    if type(arg) == type([]):
        for k in range(len(arg)):
            for i in range(N):
                for j in range(N):
                    bj = sp.Symbol('b%i'%(j,), commutative = False)
                    bdj = sp.Symbol('bd%i'%(j,), commutative = False)
                    nj = sp.Symbol('n%i'%(j,), commutative = False)

                    for g in range(N):
                        bg = sp.Symbol('b%i'%(g,), commutative = False)
                        bdg = sp.Symbol('bd%i'%(g,), commutative = False)
                        
                        arg[k][i] = arg[k][i].subs(bg*bj,0)
                        arg[k][i] = arg[k][i].subs(bdg*bdj,0)
                        
                        if j == g:
                            arg[k][i] = arg[k][i].subs(bj*bdg, bdg*bj+1)
                        else:
                            arg[k][i] = arg[k][i].subs(bj*bdg, bdg*bj)

        print "Applied: Commutation"
        return arg

# Replaces expressions arranged by apply_commutation
def replace_bdb(atom_list, arg):
    """Replaces bdq*bq with nq when q = q'"""
    N = len(atom_list)
    for k in range(len(arg)):
        for i in range(N):
            for j in range(N):
                bj = sp.Symbol('b%i'%(j,), commutative = False)
                bdj = sp.Symbol('bd%i'%(j,), commutative = False)
                nj = sp.Symbol('n%i'%(j,), real = True)

                for g in range(N):
                    bg = sp.Symbol('b%i'%(g,), commutative = False)
                    bdg = sp.Symbol('bd%i'%(g,), commutative = False)

                    if j == g:
                        arg[k][i] = (arg[k][i].subs(bdg*bj, nj))

                    elif j != g:
                        arg[k][i] = (arg[k][i].subs((bdj*bg), 0))
                        arg[k][i] = (arg[k][i].subs((bdg*bj), 0))

                    arg[k][i] = (arg[k][i].subs((bdj*bdg), 0))
                    arg[k][i] = (arg[k][i].subs((bj*bg), 0))
                    arg[k][i] = (arg[k][i].subs((bdg*nj), 0))
                    arg[k][i] = (arg[k][i].subs((bg*nj), 0))

            arg[k][i] = arg[k][i].subs(QP_SYM,Q_SYM).subs(WQP_SYM,WQ_SYM)

    print "Applied: bdq*bq Replacement"
    return arg

#def generate_cross_section_old(interactionfile, spinfile, lattice, arg, 
#                       tau_list, h_list, k_list, l_list, w_list, temp, eig_eps = 0.01):
#    """
#    Calculates the cross_section given the following parameters:
#    interactionfile, spinfile - files to get atom data
#    lattice         - Lattice object from tripleaxisproject
#    arg             - reduced list of operator combinations
#    tau_list        - list of tau position
#    h_,k_,l_lists   - lists of scan positions in (h,k,l) space.
#                    - create kappa vector with these.
#    w_list          - list of w's probed
#    temp            - temperature
#    """
#
#    # Read files, get atom_list and such
#    atom_list, jnums, jmats, N_atoms_uc = readFiles(interactionfile, spinfile)
#
#    # Get Hsave to calculate its eigenvalues
#    N_atoms = len(atom_list)
#    Hsave = calculate_dispersion(atom_list, N_atoms_uc, N_atoms, jmats, showEigs=False)
#    print Hsave
#    atom_list = atom_list[:N_atoms_uc]
#    N_atoms = len(atom_list)
##    N = N_atoms
#    
#    print "Calculated: Dispersion Relation"
#
#    # Generate kappas from (h,k,l)
#    kapvect, kaprange, kapunit = generate_kappa(lattice, h_list, k_list, w_list)
#    nkpts = len(kaprange)
#
#    # Grabs the unit vectors from the back of the lists. 
#    unit_vect = []
#    for i in range(len(arg)):
#        unit_vect.append(arg[i].pop())
#    #print unit_vect
#    unit_vect = sum(unit_vect)
#
#    unit_vect = unit_vect.subs(KAPXHAT_SYM*KAPYHAT_SYM,0)
#    unit_vect = unit_vect.subs(KAPXHAT_SYM*KAPZHAT_SYM,0)
#    unit_vect = unit_vect.subs(KAPYHAT_SYM*KAPZHAT_SYM,0)
#    unit_vect = unit_vect.subs(KAPZHAT_SYM*KAPYHAT_SYM,0)
#    unit_vect = unit_vect.subs(KAPZHAT_SYM*KAPXHAT_SYM,0)
#    unit_vect = unit_vect.subs(KAPYHAT_SYM*KAPXHAT_SYM,0)
#
#    # Generate qs from kappas and taus
##    qlist=[]
##    ones_list=np.ones((1,nkpts),'Float64')
#    
##    for tau in tau_list:
##        taui=np.ones((nkpts,3),'Float64')
##        taui[:,0]=ones_list*tau[0]
##        taui[:,1]=ones_list*tau[1]
##        taui[:,2]=ones_list*tau[2]
##        kappa_minus_tau=kapvect-taui
##        tau_minus_kappa=taui - kapvect
##               
##        qlist.append(np.vstack([kappa_minus_tau,tau_minus_kappa]))
#
#    # Eigenvalues and omegas
#    wtlist = w_list
#    print "Calculating: Eigenvalues"
#    eig_list = generate_eigenvals(Hsave, h_list, k_list, l_list, eig_eps)
#    print 'eig shape', eig_list.shape
#    print "Calculated: Eigenvalues"
#
#    # Form Factor
#    ff_list = generate_form_factors(N_atoms_uc, atom_list, kaprange)
#    print "Calculated: Form Factors"
#
#    # Generate most general form of csection
#    csection = []
#    for i in range(len(arg)):
#        for j in range(len(arg[i])):
#            csection.append(arg[i][j]*unit_vect)
#    csection = sp.Add(*csection)
#
#    for i in range(N_atoms):
#        ni = sp.Symbol('n%i'%(i,), real = True)
#        np1i = sp.Symbol('np1%i'%(i,), Real = True)
#        csection.subs(ni+1,np1i)
#
#    # start refining the cross-section
#    csection = csection.expand()
#
#    # note: xhat*yhat = xhat*zhat = yhat*zhat = 0
##    csection = csection.subs(KAPXHAT_SYM*KAPYHAT_SYM,0)
##    csection = csection.subs(KAPXHAT_SYM*KAPZHAT_SYM,0)
##    csection = csection.subs(KAPYHAT_SYM*KAPZHAT_SYM,0)
##    csection = csection.subs(KAPZHAT_SYM*KAPYHAT_SYM,0)
##    csection = csection.subs(KAPZHAT_SYM*KAPXHAT_SYM,0)
##    csection = csection.subs(KAPYHAT_SYM*KAPXHAT_SYM,0)
#
#    #  multiply by exp(-iwt)
#    # then we do the heart of the substituting which is taking 
#    # things of the form integral_{-inf}^{+inf} exp(-i(ql-wt))exp(-iwt) -> delta(k+-q-t)delta(w-wt)
#    # we just use a symbol for the delta(k+-q-t) since we don't actually evaluate them
#    # we also use a lorentzian instead of a delta function for delta(w-wt) with lifetime (1/2), adjustable at top of file)
#    csection = (csection * exp(-I * W_SYM * T_SYM) * exp(I * KAP_SYM * L_SYM)).expand(deep=False)
#    csection = sp.powsimp(csection, deep=True)
#    print 'beginning'
#    print csection
##    csection = sp.powsimp(csection)
#    csection = sub_in(csection,exp(I*T_SYM*A_WILD + I*T_SYM*B_WILD + I*C_WILD + I*D_WILD + I*K_WILD),sp.DiracDelta(A_WILD*T_SYM + B_WILD*T_SYM + C_WILD + D_WILD + K_WILD))
#    print 'intermediate'
##    print csection
##    csection = sub_in(csection,sp.DiracDelta(A*t + B*t + C*L + D*L ),(1./hbar)*sp.DiracDelta(A + B)*sp.simplify(sp.DiracDelta(C + D  - tau)))  #This is correct
#    csection = sub_in(csection,sp.DiracDelta(A_WILD*T_SYM + B_WILD*T_SYM + C_WILD*L_SYM + D_WILD*L_SYM ),sp.Pow(pi,-1)*(LIFETIME_VALUE*0.5)*sp.Pow((A_WILD+B_WILD)**2+(LIFETIME_VALUE*0.5)**2,-1)*sp.simplify(sp.DiracDelta(C_WILD + D_WILD  - TAU_SYM)))
#    print 'ending'
##    print csection
#
#    # Do some associative clean up to make it easier for later substitutions
#    csection = sub_in(csection,sp.DiracDelta(-A_WILD - B_WILD),sp.DiracDelta(A_WILD + B_WILD))
#    csection = sub_in(csection,(-A_WILD - B_WILD)**2,(A_WILD + B_WILD)**2)
#    csection = csection.subs(sp.DiracDelta(Q_SYM + TAU_SYM - KAP_SYM),sp.DiracDelta(KAP_SYM - Q_SYM - TAU_SYM))
#    csection = csection.subs(sp.DiracDelta(TAU_SYM - KAP_SYM - Q_SYM),sp.DiracDelta(KAP_SYM + Q_SYM - TAU_SYM))
#
#    csection = csection.subs(sp.DiracDelta(KAP_SYM - Q_SYM - TAU_SYM),DD_KTMQ_SYM)
#    csection = csection.subs(sp.DiracDelta(KAP_SYM + Q_SYM - TAU_SYM),DD_KTPQ_SYM)
#    print "Applied: Delta Function Conversion"
#
#    csection = csection.evalf(chop=True)
#    csection = sp.re(csection)
#
#    # SUB IN SYMBOL OR EXPRESSION
#    for i in range(N_atoms_uc):
#        ni = sp.Symbol('n%i'%(i,), real = True)
#        print ni
#        nq = Pow(sp.exp(abs(WQ_SYM)/(BOLTZ_VALUE*temp))-1,-1)
#        csection = csection.subs(ni,nq)
#
#    print csection
#
#    np.savez('csection_calc_data.npz',csection=np.array([csection]), hsave=np.array([Hsave]), ff=ff_list,)
#
#    print "Generated: Analytic Cross-Section Expression"
#    return (N_atoms_uc, csection, kaprange, tau_list, eig_list, kapvect, wtlist, ff_list)

#def chop(expr,tol=1e-8):
#    for item in expr.atoms():
#        if item < tol:
#            expr = expr.subs(item,0)
#    return expr

#@profile


#def save_cs_files(xarr,yarr,zarr,others):
#    "Takes an x, y, and z array and saves them"
#
#    file_pathname = os.path.abspath('')
#    
#    np.save(os.path.join(file_pathname,r'csx'),xarr)
#    np.save(os.path.join(file_pathname,r'csy'),yarr)
#    np.save(os.path.join(file_pathname,r'csz'),zarr)
#    
#    i=1
#    for oth in others:
#        np.save(os.path.join(file_pathname,r'cs_other_arr%i'%i),oth)
#        i=i+1
#    
#    print "Files Saved"

def generate_hkl(hkl_interval, direction):
    
    #TODO - THIS DOES NOT WORK FOR ALL CASES! only when directional components are the same or 0
    #Fixed - testing
    
    #What if other points are 0 (not the first or last points)?
    #For that to happen they would need to generate an odd number of points across an interval -x to x
    if hkl_interval[0] == 0:
        hkl_interval[0] = 1e-3
    elif hkl_interval[1] == 0:
        hkl_interval[1] = -1e-3
        
    print "hkl_interval: ", hkl_interval
    print "direction: ", direction
    
    #We should be multipliying these linspaces by the components, kx,ky,kz, where those are
    #normalized so that kx**2 + ky**2 + kz**2 = 1 (assuming that h**2+k**2+l**2=q**2), I'm not
    #entirely sure if this works in inverse space.
    norm = np.sqrt(direction['kx']**2 + direction['ky']**2 + direction['kz']**2)

    if direction['kx']:
        h_list = np.linspace(hkl_interval[0],hkl_interval[1],hkl_interval[2])*direction['kx']/norm
    else:
        h_list = np.zeros(hkl_interval[2])
    if direction['ky']:
        k_list = np.linspace(hkl_interval[0],hkl_interval[1],hkl_interval[2])*direction['ky']/norm
    else:
        k_list = np.zeros(hkl_interval[2])
    if direction['kz']:
        l_list = np.linspace(hkl_interval[0],hkl_interval[1],hkl_interval[2])*direction['kz']/norm
    else:
        l_list = np.zeros(hkl_interval[2])
        
    #print "h,k,l points: "
    #for i in range(len(h_list)):
    #    print "(",h_list[i],", ", k_list[i], ", ", l_list[i], ")"

    return h_list, k_list, l_list

def generate_wt(omega_interval):

    return np.linspace(omega_interval[0],omega_interval[1],omega_interval[2])

def generate_kappa(lattice, h,k,l):

    #Tom 2/11/11 - it seems kaprange = sqrt((ha)**2 + (kb)**2 + (lc)**2)
    kaprange=lattice.modvec(h,k,l, 'latticestar')
    nkpts=len(kaprange)

    kapvect=np.empty((nkpts,3),'Float64')
    kapvect[:,0]=h
    kapvect[:,1]=k
    kapvect[:,2]=l
    
    kapunit = kapvect.copy()
    kapunit[:,0]=kapvect[:,0]/kaprange
    kapunit[:,1]=kapvect[:,1]/kaprange
    kapunit[:,2]=kapvect[:,2]/kaprange
    
    return kapvect, kaprange, kapunit

def generate_radii(kapvect):
    rad_list=[]
    for kappa in kapvect:
        kx,ky,kz = kappa[0],kappa[1],kappa[2]
        r = np.sqrt(kx*kx+ky*ky+kz*kz)
        rad_list.append(r)
    return np.array(rad_list)



def generate_eigenvals(Hsave, h, k , l, eps):
    eig_list = calc_eigs_numerically(Hsave,h,k,l)
    # THIS PART IS SUPER IMPORTANT.
    # It makes sure that we don't consider points where the eigenvalues are close to zero.
    # Those values correspond to elastic neutron scattering which we don't want to consider.
    # Also, the entire calculation gives NaN results everywhere if this is not included. 
    eig_list = np.abs(eig_list)
    eig_list = np.where(eig_list < eps, eps, eig_list)
    return eig_list
    #return gen_eigs_with_sign(Hsave, h,k,l,eps)

def calc_eigs_numerically(mat,h,k,l,S=1):
    """
    Give it a matrix, and the (h,k,l) values to substitute into that matrix, each in a separate list.
    S is automatically evaluated as one, but can be changed. h,k,l lists must be the same length.
    """
    #get rid of these
    S_SYM = sp.Symbol('S')
    KX_SYM = sp.Symbol('kx')
    KY_SYM = sp.Symbol('ky')
    KZ_SYM = sp.Symbol('kz')        

    #lambdification functionality
    syms = (S_SYM,KX_SYM,KY_SYM,KZ_SYM)
    matsym = mat.tolist()
    func = sp.lambdify(syms,matsym,modules=["sympy"])
    
    eigarr = []
    Slist = S*np.ones(h.shape)
    
    # reduce symbolic matrix to numerical matrix and calculate the eigenvalues
    for i in range(len(h)):
        eigmat = np.array(func(Slist[i],h[i],k[i],l[i]))
        
        # Convert numpy array to sympy matrix and lambdify it to
        # exchange sympy.I with numpy's 1j. Then convert it back to 
        # a numpy array and append it to the list of eigs. 
        eigmat = sp.Matrix(eigmat)
        I2jfunc = sp.lambdify((sp.I),eigmat,modules="numpy")
        eigmat = np.array(I2jfunc(1j))

        eigs,vects = np.linalg.eig(eigmat)
        eigarr.append(eigs)
    return np.array(eigarr)

def generate_form_factors(N_atoms_uc, atom_list, hkl_interval):
    # Form Factor
    eval_pnts = np.linspace(hkl_interval[0],hkl_interval[1],hkl_interval[2])
    ff_list = []
    for i in range(N_atoms_uc):
        try:
            elSym = atom_list[i].label
            elSym = elements.symbol(elSym)
            elMass = atom_list[i].massNum
            if massNum == None: #The mass number will now generally be None -Tom 1/28/11
                el = elSym
            else:
                el = elSym[elMass]
            print el
            val = atom_list[i].valence
            if val != None:
                Mq = el.magnetic_ff[val].M_Q(eval_pnts)
            else:
                Mq = el.magnetic_ff[0].M_Q(eval_pnts)
        except:
            Mq = np.zeros(len(eval_pnts))
        #Test to see how things look without ff
        #Mq = np.ones(len(eval_pnts))
        
        ff_list = Mq
#    fake = np.ones(len(kaprange))
    return np.array(ff_list)

def generate_cross_section(arg, N, N_atoms_uc):
    unit_vect = []
    for i in range(len(arg)):
        unit_vect.append(arg[i].pop())
    #print unit_vect
    unit_vect = sum(unit_vect)

    unit_vect = unit_vect.subs(KAPXHAT_SYM*KAPYHAT_SYM,0)
    unit_vect = unit_vect.subs(KAPXHAT_SYM*KAPZHAT_SYM,0)
    unit_vect = unit_vect.subs(KAPYHAT_SYM*KAPZHAT_SYM,0)
    unit_vect = unit_vect.subs(KAPZHAT_SYM*KAPYHAT_SYM,0)
    unit_vect = unit_vect.subs(KAPZHAT_SYM*KAPXHAT_SYM,0)
    unit_vect = unit_vect.subs(KAPYHAT_SYM*KAPXHAT_SYM,0)

    # Generate most general form of csection
    csection = []
    for i in range(len(arg)):
        for j in range(len(arg[i])):
            csection.append(arg[i][j]*unit_vect)
    csection = sp.Add(*csection)
    
    print "Current Csection:\n", csection
                
    for i in range(N):
        ni = sp.Symbol('n%i'%(i,), real = True)
        np1i = sp.Symbol('np1%i'%(i,), Real = True)
        csection.subs(ni+1,np1i)

    # start refining the cross-section
    print "Csection before expansion:\n", csection
    csection = csection.expand()

    # note: xhat*yhat = xhat*zhat = yhat*zhat = 0
#    csection = csection.subs(KAPXHAT_SYM*KAPYHAT_SYM,0)
#    csection = csection.subs(KAPXHAT_SYM*KAPZHAT_SYM,0)
#    csection = csection.subs(KAPYHAT_SYM*KAPZHAT_SYM,0)
#    csection = csection.subs(KAPZHAT_SYM*KAPYHAT_SYM,0)
#    csection = csection.subs(KAPZHAT_SYM*KAPXHAT_SYM,0)
#    csection = csection.subs(KAPYHAT_SYM*KAPXHAT_SYM,0)
        
    #  multiply by exp(-iwt)
    # then we do the heart of the substituting which is taking 
    # things of the form integral_{-inf}^{+inf} exp(-i(ql-wt))exp(-iwt) -> delta(k+-q-t)delta(w-wt)
    # we just use a symbol for the delta(k+-q-t) since we don't actually evaluate them
    # we also use a lorentzian instead of a delta function for delta(w-wt) with lifetime (1/2), adjustable at top of file)
    
    #Bill's line:
    #csection = (csection * exp(-I * W_SYM * T_SYM) * exp(I * KAP_SYM * L_SYM)).expand(deep=False)
    #Addition of e^iQ.R term
    print "Csection before adding the L term:\n", csection
    csection = (csection * exp(-I * W_SYM * T_SYM) * exp(I * KAP_SYM * (L_SYM))).expand(deep=False)
    print "\ncsection after adding L term:\n", csection
    
    #Tom - 2/4/11 Adding e^iQ.R term where is the basis vector
    #csection = csection*exp(I*QR_DOT_SYM)

    csection = sp.powsimp(csection, deep=True)
    print 'beginning'
    print csection
    #csection = sp.powsimp(csection)
    #TOM 2/11/11 - TODO - the e^iK.L term is put into the Dirac delta even though there is no T dependance.
    csection = sub_in(csection,exp(I*T_SYM*A_WILD + I*T_SYM*B_WILD + I*C_WILD + I*D_WILD + I*K_WILD),sp.DiracDelta(A_WILD*T_SYM + B_WILD*T_SYM + C_WILD + D_WILD + K_WILD))
    print 'intermediate'
    print csection
#    csection = sub_in(csection,sp.DiracDelta(A*t + B*t + C*L + D*L ),(1./hbar)*sp.DiracDelta(A + B)*sp.simplify(sp.DiracDelta(C + D  - tau)))  #This is correct
    csection = sub_in(csection,sp.DiracDelta(A_WILD*T_SYM + B_WILD*T_SYM + C_WILD*L_SYM + D_WILD*L_SYM ),sp.Pow(pi,-1)*(LIFETIME_VALUE*0.5)*sp.Pow((A_WILD+B_WILD)**2+(LIFETIME_VALUE*0.5)**2,-1)*sp.simplify(sp.DiracDelta(C_WILD + D_WILD  - TAU_SYM)))
    print 'ending' 
    print csection
    
    # Do some associative clean up to make it easier for later substitutions
    csection = sub_in(csection,sp.DiracDelta(-A_WILD - B_WILD),sp.DiracDelta(A_WILD + B_WILD))
    csection = sub_in(csection,(-A_WILD - B_WILD)**2,(A_WILD + B_WILD)**2)
    csection = csection.subs(sp.DiracDelta(Q_SYM + TAU_SYM - KAP_SYM),sp.DiracDelta(KAP_SYM - Q_SYM - TAU_SYM))
    csection = csection.subs(sp.DiracDelta(TAU_SYM - KAP_SYM - Q_SYM),sp.DiracDelta(KAP_SYM + Q_SYM - TAU_SYM))

    csection = csection.subs(sp.DiracDelta(KAP_SYM - Q_SYM - TAU_SYM),DD_KTMQ_SYM)
    csection = csection.subs(sp.DiracDelta(KAP_SYM + Q_SYM - TAU_SYM),DD_KTPQ_SYM)
    print "Applied: Delta Function Conversion\n"
    print csection
    
    #csection = csection.evalf(chop=True)
    csection = sp.re(csection)
    
    # SUB IN SYMBOL OR EXPRESSION
    for i in range(N_atoms_uc):
        ni = sp.Symbol('n%i'%(i,), real = True)
        print ni
        nq = Pow(sp.exp(abs(WQ_SYM)/(BOLTZ_VALUE*TEMP_SYM))-1,-1)
        csection = csection.subs(ni,nq)
    
    print csection
    
    print "Generated: Analytic Cross-Section Expression"
    return csection

def single_cross_section_calc(theta, phi, rad, atom_list, csection, tau, eigval, wt,
                       temperature, ffval, eief = True, efixed = 14.7, eps = 0.001):
    """
    This method uses the core of eval_cross_section. It takes a single value for wt, tau. Instead of kx,ky,kz lists,
    it takes a single of the following: theta, phi and radius. With them, it calculates kx, ky, kz and kappa. 
    
    Inputs:
    theta, phi, rad         - single values of each. used to create kx,ky,kz and kappa vector. 
    N_atoms_uc, atomlist,   - taken from readfiles(interactions,spins)
    csection                - takes the cross-section expression generated in generate_cross_section
    tau                     - single value for tau
    eig_list                - list of eigenvalues. calculated in generate_cross_section using spinwavecalcfile methods
    wt                      - single value of omega
    temperature             - temperature
    ffval                   - form factor term, taken from list of form factors calculated in generate_cross_section
    eief, efixed            - determines whether we want Ef/Ei. if true: ef/ei ef=ei-eigenval and ei=14.7 meV
    eps                     - this value determines the size of the epsilon neighborhood around 0 with which is used
                              to exclude values of omega which will return an infinite thermal averaged n_q value. 
    
    This method is faster than the eval_cross_section method. Use this one.
    
    """

    # calculate front constant
    front_constant = ((GAMMA_R0_VALUE**2)*DEBYE_WALLER_VALUE/(2*pi*HBAR_VALUE)).evalf()

    # calculate kx,ky,kz and kappa
    #TOM 2/11/11 -TODO- He calculates r,theta,phi specifically for this function... why does he then convert them right back?
    #and are these kx,ky,kz = kappa = h,k,l?  because the "kappa" he creates here is not equal to (h,k,l) because of the a,b,c factor he put into the radius
    #in the Lattice class
    kx = rad*sp.sin(theta)*sp.cos(phi)
    ky = rad*sp.sin(theta)*sp.sin(phi)
    kz = rad*sp.cos(theta)
    kap = array([kx,ky,kz])   

    # subs in kappa unit vectors
    csection = csection.subs(KAPXHAT_SYM,kx/rad)
    csection = csection.subs(KAPYHAT_SYM,ky/rad)
    csection = csection.subs(KAPZHAT_SYM,kz/rad)

    #make two copies of the cross-section, one + and one -
    csectempp = copy(csection)  #good
    csectempm = copy(csection)  #good     
    
    # get rid of the appropriate delta(k-t+-q) factor
    csectempp = csectempp.subs(DD_KTMQ_SYM,sp.S(1))#not used (overwritten) ~TOM
    csectempp = csectempp.subs(DD_KTPQ_SYM,sp.S(0))
    csectempm = csectempm.subs(DD_KTMQ_SYM,sp.S(0))#not used (overwritten) ~TOM
    csectempm = csectempm.subs(DD_KTPQ_SYM,sp.S(1))

    # sum over all the different eigenvalues
    csdata = csectempp+csectempm
    print csdata

    cs_func = sp.lambdify((THETA_SYM, PHI_SYM, W_SYM, WQ_SYM, TEMP_SYM), csdata, modules="sympy")

    csdata = cs_func(theta, phi, wt, eigval, temperature)

    # get the kp/k factor
    # eief == True => ei=efixed
    # eief == False => ef=efixed    
    if eief:
        ei = efixed
#            ef = ei - eigi
        ef = ei - eigval
    else:
        ef = efixed
#            ef = ei + eigi
        ei = ef + eigval
    kpk = ef/ei
    
    #Multiply data by front constants
    csdata = front_constant*kpk*csdata
    
    # Multiply by form factor
    csdata = (0.5*G_VALUE*ffval)**2*csdata
    
    print "completed first time"
    print csdata
    return csdata


def spherical_averaging(rad, wt, eigval, tau, ffval, N_atoms_uc, atom_list, csection,
                       temperature, eief = True, efixed = 14.7,thetas=None,phis=None,):
    """
    This method takes a single radius, omega and tau and calculates the spherical average at that point
    using the single_cross_section_calc method.
    
    N_atoms_uc - number of atoms in unit cell
    csection - analytic cross-section expression
    kap - kappa vector
    tau - tau
    eig_list - list of eigenvalues
    wt - omega
    ffval - form factor value
    temperature - temperature
    eief - True => E_initial = efixed, False => E_final = efixed
    efixed - fixed energy; either E_final or E_initial, subject to eief
    """
    
#    args=(atom_list, csection, tau, eigval, wt, temperature, ffval, eief, efixed)#ffval, eief, efixed)

#    theta_test=np.pi/2.0
#    phi_test = np.pi/4.0

    """
    NEEDED FOR DISTRIBUTED COMPUTATION
    """
    THETA_SYM = sp.Symbol('theta', real = True)
    PHI_SYM = sp.Symbol('phi', real = True)
    TAU_SYM = sp.Symbol('tau', real = True)
    EIG_SYM = sp.Symbol('eig', real = True)
    WT_SYM = sp.Symbol('wt', real = True)
    RAD_SYM = sp.Symbol('rad', real = True)
    FF_SYM = sp.Symbol('ff', real = True)

    if thetas == None:
        thetas = np.linspace(0,np.pi,25)
    if phis == None:
        phis = np.linspace(0,2*np.pi,25)
    cs_vals = []
    partial_res=[]

    start1 = clock()

    val_expr = single_cross_section_calc(THETA_SYM, PHI_SYM, RAD_SYM, atom_list, csection, TAU_SYM,
                                         EIG_SYM, WT_SYM, temperature, FF_SYM, eief, efixed)*sp.sin(THETA_SYM)*RAD_SYM**2
    print val_expr
    val_func = sp.lambdify((THETA_SYM, PHI_SYM, RAD_SYM, TAU_SYM, EIG_SYM, WT_SYM, FF_SYM), 
                           val_expr, modules = "sympy")

    cs_vals = np.array([[val_func(t, p, rad, tau, eigval, wt, 1.) for p in phis] for t in thetas])

    print 'done once'

    end1 = clock()
    calc_time = end1-start1

    start2 = clock()
    partial_res = np.array([simps(cs_vals[i],phis) for i in range(len(thetas))])
#    for i in range(len(thetas)):
#        single_res = simps(cs[i],phis)
#        partial_res.append(single_res)
    total_res = simps(partial_res,thetas)
    end2 = clock()
    inte_time = end2-start2

    print 'result', #total_res
    print 'calc time', calc_time
    print 'inte time', inte_time
    return total_res

def run_cross_section(interactionfile, spinfile, tau_list, temperature, 
                      direction={'kx':1,'ky':0,'kz':0}, hkl_interval=[1e-3,2*np.pi,1000], 
                      omega_interval=[0,5,1000], eig_eps = 0.01, 
                      atom_list = None, N_atoms_uc = None, Hsave = None, nosave=False):
    """
    We use this method to generate the expression for the cross-section given just the interaction and spins file.
    *** Use this first to calculate cross-section before using any other methods as they all need the csection expression
    this generates.***
    
    I have the infrastructure in place for this "steps" and bounds for the scan like the dispersion calc has:
    direction       - of the form [h,k,l] where h,k,l are either 0 or 1. For example, a scan along h would be
                    [1,0,0].
    hkl_interval    - of the form [start,end,#of steps].
    omega_interval  - of the form [start,end,#of steps].
    """
    
    #mcQueeny_case(interactionfile, spinfile, tau_list, temperature, direction, hkl_interval, 
    #              omega_interval, eig_eps)
    
    
    start = clock()

    # Generate Inputs
    if not atom_list or not N_atoms_uc:
        atom_list, jnums, jmats, N_atoms_uc = readFiles(interactionfile,spinfile)
    N_atoms = len(atom_list)
    if not Hsave:
        Hsave = calculate_dispersion(atom_list,N_atoms_uc,N_atoms,jmats,showEigs=False)

    atom_list=atom_list[:N_atoms_uc]

    k = spm.Matrix([KX_SYM,KY_SYM,KZ_SYM])

    (b,bd) = generate_b_bd_operators(atom_list)
    (a,ad,a0,ad0) = generate_a_ad_operators(atom_list, k, b, bd)
    (Sp,Sm,Sp0,Sm0) = generate_Sp_Sm_operators(atom_list, a, ad, a0, ad0)
    (Sa,Sb,Sn,Sa0,Sb0,Sn0) = generate_Sa_Sb_Sn_operators(atom_list, Sp, Sm, Sp0, Sm0)
    (Sx,Sy,Sz,Sx0,Sy0,Sz0) = generate_Sx_Sy_Sz_operators(atom_list, Sa, Sb, Sn, Sa0, Sb0, Sn0)
    list_print(Sx)
    print ''

    #Ham = generate_Hamiltonian(N_atoms, atom_list, b, bd)
    ops = generate_possible_combinations(atom_list, [Sx,Sy,Sz], [Sx0,Sy0,Sz0])
    ops = holstein(atom_list, ops)
    ops = apply_commutation(atom_list, ops)
    ops = replace_bdb(atom_list, ops)
    ops = reduce_options(atom_list, ops)
    
    print "prelims complete. generating cross-section","\n"
    
    lattice = generate_lattice()
    h_list, k_list, l_list = generate_hkl(hkl_interval, direction)    
    wt_list = generate_wt(omega_interval)
    
    
    #run fortran code with the same parameters for testing
    #f_output(h_list, k_list, l_list, wt_list, temperature)

    kapvect, kaprange, kapunit = generate_kappa(lattice, h_list, k_list, l_list)

    # Eigenvalues and omegas
    print "Calculating: Eigenvalues"
    eig_list = generate_eigenvals(Hsave, h_list, k_list, l_list, eig_eps)
    print 'eig shape', eig_list.shape
    print "Calculated: Eigenvalues"

    # Form Factor
    fflist = generate_form_factors(N_atoms_uc, atom_list, hkl_interval)
    print "Calculated: Form Factors"

    csection = generate_cross_section(ops, N_atoms, N_atoms_uc)

    end = clock()
    print 'Generatation Time', end-start
    
    if not nosave:
        np.savez('csection_calc_data.npz',csection=np.array([csection]), hsave=np.array([Hsave]))
    
    return atom_list,N_atoms_uc,csection,kaprange,tau_list,eig_list,kapvect,wt_list,fflist
            
                                     

def gen_eigs_with_sign(Hsave, h, k , l, eps):
    """This is just Bill's method, generate_eigenvals except it doesn't take the absolute value."""
    eig_list = calc_eigs_numerically(Hsave,h,k,l)
    # THIS PART IS SUPER IMPORTANT.
    # It makes sure that we don't consider points where the eigenvalues are close to zero.
    # Those values correspond to elastic neutron scattering which we don't want to consider.
    # Also, the entire calculation gives NaN results everywhere if this is not included.
    signs = eig_list/np.abs(eig_list)
    eig_list = np.abs(eig_list)
    eig_list = np.where(eig_list < eps, eps, eig_list)
    return eig_list*signs

def calc_eig_mats_numerically(mat,h,k,l,S=1):
    """
    Copy of calc_eigs_numerically that instead returns the vectors rather than the eigenvalues(for testing).
    """
    #get rid of these
    S_SYM = sp.Symbol('S')
    KX_SYM = sp.Symbol('kx')
    KY_SYM = sp.Symbol('ky')
    KZ_SYM = sp.Symbol('kz')        

    #lambdification functionality
    syms = (S_SYM,KX_SYM,KY_SYM,KZ_SYM)
    matsym = mat.tolist()
    func = sp.lambdify(syms,matsym,modules=["sympy"])
    
    eigarr = []
    matarr = []
    Slist = S*np.ones(h.shape)
    
    # reduce symbolic matrix to numerical matrix and calculate the eigenvalues
    for i in range(len(h)):
        eigmat = np.array(func(Slist[i],h[i],k[i],l[i]))
        
        # Convert numpy array to sympy matrix and lambdify it to
        # exchange sympy.I with numpy's 1j. Then convert it back to 
        # a numpy array and append it to the list of eigs. 
        eigmat = sp.Matrix(eigmat)
        I2jfunc = sp.lambdify((sp.I),eigmat,modules="numpy")
        eigmat = np.array(I2jfunc(1j))

        eigs,vects = np.linalg.eig(eigmat)
        eigarr.append(vects)
        matarr.append(eigmat)
    return np.array(eigarr), matarr  


def run_eval_pointwise(N_atoms_uc,atom_list,csection,kaprange,tau_list,eig_list,kapvect,wtlist,fflist,temp,direction):
    """
    This method uses the single_cross_section_calc method to evaluate the csection expression. We have the method
    wrapped in some for loops and it appears faster and more precise than run_eval_cross_section. 
    
    *** USE THIS AFTER generate_cross_section ***
    
    Takes values that are returned from run_cross_section. Returns values to be used to plot results - basically: x,y,z
    where x.shape = (n,0), y.shape = (m,0) and z.shape = (n,m)
    """
    
    h_list = kapvect[:,0]
    k_list = kapvect[:,1]
    l_list = kapvect[:,2]
    w_list = wtlist
    temperature = temp
    
    #Tom - 2/11/11 TODO - Is this right?   He has a factor of (a,b,c) in kaprange, but not h,k,l
    rad_list = np.array(kaprange)
    theta_list = np.array(np.arccos(l_list/rad_list))
    phi_list = np.array(arctan2(k_list,h_list))
    
    tau_list = np.array(tau_list)
    w_list = np.array(w_list)
    
    print "Computing Numerical Evaluation of Cross-section"

#    THETA,PHI,RAD,FF_SYM = sp.Symbol('THETA'),sp.Symbol('PHI'),sp.Symbol('RAD'),sp.Symbol('ff')

    print 'Generating Expression'
    
    csection_func_expr = single_cross_section_calc(THETA_SYM ,PHI_SYM, RAD_SYM, atom_list, csection, TAU_SYM, EIG_SYM,
                                                          WT_SYM, TEMP_SYM, FF_SYM)
    
    print 'CCC',csection_func_expr
    print 'Evaluating Expression'
    
    #lambdify makes this a function that takes arguments in the order given ~Tom
    csection_func = sp.lambdify((THETA_SYM ,PHI_SYM ,RAD_SYM ,TAU_SYM, EIG_SYM, WT_SYM, TEMP_SYM, FF_SYM),csection_func_expr, modules="numpy")
    
    print 'Generating Array'
    
    values = csection_func(theta_list[:,nax,nax,nax],phi_list[:,nax,nax,nax],rad_list[:,nax,nax,nax],
                           tau_list[nax,nax,:,nax],eig_list[:,:,nax,nax],wtlist[nax,nax,nax,:],temp, fflist[:,nax,nax,nax]).sum(axis=1).sum(axis=1)
    
    print 'Complete'
#    print values.shape
    
    x_list = []
    #Bill's code:
    #if direction:
    #    if direction['kx']: x_list = h_list
    #    elif direction['ky']: x_list = k_listrun_eval_p
    #    else: x_list = l_list
    #else: x_list = h_list
    #Tom's:
    x_list = np.sqrt(h_list**2 + k_list**2 + l_list**2) #element by element
  
    return x_list, w_list, values.T

def run_spherical_averaging(N_atoms_uc,atom_list,csection,kapvect,tau_list,eig_list,wt_list,fflist,temperature):
    """
    This method runs the spherical averaging method to calculate the spherically averaged scattering cross_section. 
    
    Currently, results are iffy and slow. We are working on compatiability with compufans to speed things up. We could
    also use pyro to take advantage of multiple cores on a single machine. 
    
    """
    rad_list = generate_radii(kapvect)
    lattice = generate_lattice()
    
    res_array = []
#    rand_wt_list = np.append(wt_list[::2],wt_list[1::2])
    xvals = np.array(rad_list)
    yvals = np.array(wt_list)
    
    tau_list = np.array(tau_list)
    wt_list = np.array(wt_list)
    rad_list = np.array(rad_list)
    eig_list = np.array(eig_list)
    fflist = np.array(fflist)
#    cs_vals = []
#    partial_res=[]

    expr = spherical_averaging(RAD_SYM, W_SYM, EIG_SYM, TAU_SYM, FF_SYM, N_atoms_uc, atom_list, csection, temperature)
    func = sp.lambdify((RAD_SYM, W_SYM, EIG_SYM, TAU_SYM, FF_SYM,), expr, modules = "numpy")
    print "Generating Array"
    arr_st = clock()
    res_array = func(rad_list[:,nax,nax,nax], wt_list[nax,nax,nax,:], eig_list[:,:,nax,nax],
                       tau_list[nax,nax,:,nax], 1.).sum(axis=2).sum(axis=1)
    res_array = res_array *(fflist[:,None]*fflist[None,:])
    arr_en = clock()
    print "Array Formed in", arr_en-arr_st
    print res_array.shape    

    return xvals,yvals,np.fliplr(res_array.T)


def plot_cross_section(xi, wtlist, csdata, colorbarFlag = True, minval = 0, maxval = 25):
    """
    Used for plotting. Needs to be more robust before release.
    Also, watch the limits for the intensity contours. Something is tricky with it and
    doesn't want to show values that are close to 0 as different from 0. 
    
    """
    xi = xi # kapvect[:,0]
    yi = wtlist
    zi = np.array(csdata,'Float64')
    
    #test
    #for i in range(len(xi)):
    #    print xi[i], " ", yi[i], " ", zi[i], "\n"

    zmin, zmax = np.min(zi), np.max(zi)
    if zmin < minval or zmin == sp.nan:
        print 'plotting clipped minimal extrema'
        zi = np.where(zi < minval, minval, zi)
#        zi = np.where(zi != sp.nan, minval, zi)
    if zmax > maxval or zmax == sp.nan:
        print 'plotting clipped maximal extrema'
        zi = np.where(zi > maxval, maxval, zi)
#        zi = np.where(zi != sp.nan, maxval, zi)
    print zmin, zmax

    if colorbarFlag:
        locator = ticker.MaxNLocator(25)
        locator.create_dummy_axis()
        locator.set_bounds(minval, maxval)
        levs = locator()
        levs[0]=0.5
        plt.contourf(xi,yi,zi, levs)
        
        l_f = ticker.LogFormatter(10, labelOnlyBase=False)
        cbar = plt.colorbar(ticks = levs, format = l_f)
    else: 
        plt.contourf(xi,yi,zi)
        cbar = plt.colorbar()
    
    plt.show()
    


#def correction_driver(interactionfile, spinfile, tau_list, temperature, 
#                      direction=[1,0,0], hkl_interval=[1e-3,2*np.pi,1000], omega_interval=[0,5,1000]):
#    import csection_calc as csc
#    
#    atom_list, jnums, jmats,N_atoms_uc=readFiles(interactionfile,spinfile)
#    
#    atom_list=atom_list[:N_atoms_uc]
#    N_atoms = len(atom_list)
#
#    k = spm.Matrix([KX_SYM,KY_SYM,KZ_SYM])
#    
#    (b,bd) = generate_b_bd_operators(atom_list)
#    (a,ad,a0,ad0) = generate_a_ad_operators(atom_list, k, b, bd)
#    (Sp,Sm,Sp0,Sm0) = generate_Sp_Sm_operators(atom_list, a, ad, a0, ad0)
#    (Sa,Sb,Sn,Sa0,Sb0,Sn0) = generate_Sa_Sb_Sn_operators(atom_list, Sp, Sm, Sp0, Sm0)
#    (Sx,Sy,Sz,Sx0,Sy0,Sz0) = generate_Sx_Sy_Sz_operators(atom_list, Sa, Sb, Sn, Sa0, Sb0, Sn0)
#
#    ops = generate_possible_combinations(atom_list, [Sx,Sy,Sz], [Sx0,Sy0,Sz0])
#    ops = holstein(atom_list, ops)
#    ops = apply_commutation(atom_list, ops)
#    ops = replace_bdb(atom_list, ops)
#    ops1 = reduce_options(atom_list, ops)
#    ops2 = csc.reduce_options(atom_list, ops)
#
#    ops1 = np.array(ops1)
#    ops2 = np.array(ops2)
#    print ops1-ops2
#    print ops1
#    print ops2

def save_arrays(outpath, x, y, z):
    np.savez(outpath, x=x, y=y, z=z)
    #test
    #for i in range(0,len(x)):
    #    print x[i],' ', y[i], ' ',z[i], '\n'
    

def load_plot(outfile, colorbarFlag=True, minval=0, maxval=25):
    print outfile, colorbarFlag, minval, maxval
    arr = np.load(outfile)
    x = arr['x']
    y = arr['y']
    z = arr['z']
    
    plot_cross_section(x, y, z, colorbarFlag = colorbarFlag, minval = minval, maxval = maxval)
    
def f_output(h,k,l,w, temperature):
    """This funtion will create the input (.dat file) for MacQueeny and Yang Zhao.'s 
    Fortran Code to compare output.  -Tom"""
    #For now I will hard code the paths
    path = '/home/tom/Desktop/spinwaves_fortran/SPINAL_MACS/'
    sw_macs_path = 'sw_macs'
    datFile = 'ferro.dat'
    parFile = 'ferro.par'
    outputFile = 'ferro.out'
    inpFile = 'ferro.inp'
    
    #create the input file
    f = open(path+inpFile, 'w')
    f.write(parFile + '\n')
    f.write(datFile + '\n')
    f.write(str(len(h)) + '\n')
    f.write(str(temperature) + '\n')
    f.write(outputFile + '\n')
    f.close()

    #create the dat file
    #format: (dc = don't care, ? is not sure)
    #
    #h k l w cnt err qx qy r0 eres ga gb gc a4in
    #h k l w dc  dc  h? k? dc ?    ?  ?  ?  dc
    f = open(path+datFile, 'w')
    for i in range(0, len(h)):
        f.write(str(h[i])+" "+str(k[i])+" "+str(l[i])+" "+str(w[i])+" 0 0 "+str(h[i])+' '+ 
                str(k[i])+' 1 1 1 1 1 1\n')
    f.close()
    #call the fortran program
    #subprocess.call(sw_macs_path + " < " + inpFile)
    os.system(path+sw_macs_path + " < " + path+inpFile)
    #sw_macs = subprocess.Popen(sw_macs_path, stdin = subprocess.PIPE, shell = True)
    #sw_macs.communicate(input = parFile)
    #sw_macs.communicate(input = datFile)
    #sw_macs.communicate(len(h))
    
    

def cs_driver(interfile, spinfile, hkl_interval, w_interval, tau_list, direction, 
              temperature, outpath, sphavg_bool, plotchars):
    """
    """
    ST = clock()
    
    plot_min, plot_max, colorbar_bool, replot_bool = plotchars
    
    if not replot_bool:
        print "\n\n\n\n\nhere\n\n\n\n"
        (atom_list, N_atoms_uc, csection, kaprange, tau_list, 
        eig_list, kapvect, wt_list, fflist) = run_cross_section(interfile, spinfile, tau_list, temperature,
                                                               direction, hkl_interval, w_interval)
    else:
        atom_list, jnums, jmats, N_atoms_uc = readFiles(interfile,spinfile)
        csection_arrays = np.load('csection_calc_data.npz')
        csection = csection_arrays['csection'][0]
        Hsave = csection_arrays['hsave'][0]
        
        lattice = generate_lattice()
        h,k,l = generate_hkl(hkl_interval,direction)
        wt_list = generate_wt(w_interval)
        kapvect, kaprange, kapunit = generate_kappa(lattice,h,k,l)
        eig_list = generate_eigenvals(Hsave,h,k,l,0.001)
        fflist = generate_form_factors(N_atoms_uc, atom_list, hkl_interval)
        
        #Generate output to compare with fortran code
        f_output(h,k,l,wt_list, temperature)

    st = clock()

    #Regular cross-section calc
    if not sphavg_bool:
        x,y,z = run_eval_pointwise(N_atoms_uc, atom_list, csection, kaprange, tau_list,
                                   eig_list, kapvect, wt_list, fflist, temperature, direction)
    else:
        x,y,z = run_spherical_averaging(N_atoms_uc, atom_list, csection, kapvect, tau_list,
                                        eig_list, wt_list, fflist, temperature)
    
    en = clock()
    print 'Evaluation Run Time', en-st

    EN = clock()
    print 'Total Run Time', EN - ST

    save_arrays(outpath, x, y, z)
    
    return outpath, colorbar_bool, plot_min, plot_max 


#---------------- MAIN --------------------------------------------------------- 

if __name__=='__main__':
    #interactionfile = "/media/7F3E-B255/NIST/Mn-anti-ferro_export.txt"
    #spinfile = "/media/7F3E-B255/NIST/Mn-z_ani-antiferro_spins_perfect.txt"
    #interactionfile = "/media/7F3E-B255/NIST/Mn-anti-ferro_body-centercubic_export.txt"
    #spinfile = "/media/7F3E-B255/NIST/Mn-anti-ferro_body-centercubic_z-spins.txt"
    #interactionfile = "/home/tom/Documents/code/Mn_cubic(sg221)2atoms_uc(a_dir)_anti-ferro_export.txt"
    #spinfile = "/home/tom/Documents/code/Mn_cubic(sg221)_anti-ferro_z-spins.txt"
    #interactionfile = "/home/tom/Documents/code/simple AF chain export(2atoms).txt"
    #spinfile = "/home/tom/Documents/code/simple AF chain spins(2atoms).txt"
    interactionfile = "/home/tom/Documents/code/15a32iexport.txt"
    spinfile = "/home/tom/Documents/code/15a32ispins.txt"
    
    
#    atom_list, jnums, jmats, N_atoms_uc = readFiles(interactionfile,spinfile)
#    N_atoms = len(atom_list)
#    Hsave = calculate_dispersion(atom_list,N_atoms_uc,N_atoms,jmats,showEigs=False)
    #to get the values in Mcqueenies paper, use  hkl_interval=[np.pi/4.0,np.pi/4.0,1]
    #and tau = [2pi,2pi,2pi] and [4*np.pi, 4*np.pi, 4*np.pi]
    mcQueeny_case(interactionfile, spinfile, [0, 0, 0], 0.001, direction={'kx':1,'ky':0,'kz':0},
                  hkl_interval=[0.001,4*np.pi,100], omega_interval=[0,5,100])                  
                  #hkl_interval=[np.pi/8.0,np.pi/8.0,1])
                  #hkl_interval=[np.pi/4.0,np.pi/4.0,1])
    
    
    if 0:
    #def pd():
        #from spinwaves.cross_section.csection_calc import spherical_averaging as sph
        
        ST = clock()
        
        file_pathname = os.path.abspath('')
        print file_pathname
        if 0: # YANG
            spinfile=r'C:/Documents and Settings/wflynn/Desktop/spins.txt'
            interfile=r'C:/Documents and Settings/wflynn/Desktop/yang_montecarlo.txt'
        if 0: # SQUARE
            spinfile=r'C:/Documents and Settings/wflynn/Desktop/spinwave_test_spins.txt'#'C:/eig_test_Spins.txt'
            interfile=r'C:/Documents and Settings/wflynn/Desktop/spinwave_test_montecarlo.txt'
        if 1: # CHAIN
            spinfile=r'C:/test_s.txt'
            interfile=r'C:/test_m.txt'
        if 0: #second CHAIN
            spinfile=r'C:/1_s.txt'
            interfile=r'C:/1_m.txt'
        temperature = 0.001
    
        tau_list = [np.array([0,0,0])]
    
        out = cs_driver(interfile, spinfile, [0,2*np.pi,100], [0,5,100], tau_list,
                  {'kx':1,'ky':0,'kz':0}, temperature, 'C:/arr.npz', False, (0,25,True,False))
        load_plot(*out)
        sys.exit()
    
        atom_list, jnums, jmats,N_atoms_uc=readFiles(interfile,spinfile)
        
    #    correction_driver(interfile,spinfile,tau_list,temperature)
    #    sys.exit()
        
        N_atoms_uc,csection,kaprange,tau_list,eig_list,kapvect,wt_list,fflist = run_cross_section(interfile,spinfile,tau_list,temperature)
    #    left_conn, right_conn = Pipe()
    #    p = Process(target = create_latex, args = (right_conn, csection, "Cross-Section"))
    #    p.start()
    #    eig_frame = LaTeXDisplayFrame(self.parent, p.pid, left_conn.recv(), 'Cross-Section')
    #    self.process_list.append(p)
    #    p.join()
    #    p.terminate()
    
        h_list = kapvect[:,0]
        k_list = kapvect[:,1]
        l_list = kapvect[:,2]
        w_list = wt_list
        
        #    syms = (DD_KTPQ_SYM,DD_KTMQ_SYM,W_SYM,WQ_SYM,KAPXHAT_SYM,KAPYHAT_SYM,KAPZHAT_SYM)
        #    vals = (1.0,       1.0        ,W_SYM,WQ_SYM,KAPXHAT_SYM,KAPYHAT_SYM,KAPZHAT_SYM)
        #    cs_func = lambdify_expr(csection,syms,vals)
        #
        #    print cs_func(W_SYM,WQ_SYM,KAPXHAT_SYM,KAPYHAT_SYM,KAPZHAT_SYM)
        #    sys.exit()
    
        # FASTER/MORE-ACCURATE METHOD TO GENERATE CROSS SECTION
        if 0:
            st = clock()
            x,y,z=run_eval_pointwise(N_atoms_uc,atom_list,csection,kaprange,tau_list,eig_list,kapvect,wt_list,fflist,temperature)
            en = clock()
            print 'Evaluation Run Time', en-st
            
            EN = clock()
            print 'Total Run Time', EN - ST
            np.save(os.path.join(file_pathname,r'myfilex.txt'),x)
            np.save(os.path.join(file_pathname,r'myfiley.txt'),y)
            np.save(os.path.join(file_pathname,r'myfilez.txt'),z)
            plot_cross_section(x,y,z)
            clear_cache()
            sys.exit()
    
        # ORIGINAL METHOD TO GENERATE CROSS SECTION
        if 0:
            st = clock()
            kapvect,wt_list,csdata=run_eval_cross_section(N_atoms_uc,csection,kaprange,tau_list,eig_list,kapvect,wt_list,fflist)
            
            en = clock()
            print en-st
            plot_cross_section(kapvect[:,0],wt_list,csdata)
            sys.exit()
        
        # TEST SINGLE VALUE SPHERICAL_AVERAGING
        if 0:
            radius = 0.2
            ffval = 0.9
            same_args = (N_atoms_uc, atom_list, csection, eig_list, temperature)
            vals=[]
            for wvalue in w_list:
                val=spherical_averaging(radius , wvalue, tau_list[0], ffval,*same_args)
                vals.append(val)
                print 'wvalue', wvalue, 'val',val
            print vals
            sys.exit()
        
        # TEST FOR SINGLE_CROSS_SECTION_CALC
        if 0:
            st = clock()
            aa = bb = cc = np.array([2.0*np.pi], 'Float64')
            alpha = beta = gamma = np.array([np.pi/2.0], 'Float64')
            vect1 = np.array([[1,0,0]])
            vect2 = np.array([[0,0,1]])
            lattice = Lattice(aa, bb, cc, alpha, beta, gamma, Orientation(vect1, vect2))
            
            ffval = 0.9
            rad_list = kaprange
            theta_list = np.arccos(l_list/rad_list)
            phi_list = np.arctan2(k_list,h_list)
            
            same_args = (N_atoms_uc, atom_list, csection, eig_list, temperature)
            
            final_vals=[]
            for wti in wt_list:
                inner_vals=[]
                for i in range(len(rad_list)):
                    inner_vals2=[]
                    for tau in tau_list:
    
                        val = single_cross_section_calc(theta_list[i], phi_list[i], rad_list[i], N_atoms_uc, atom_list, csection, tau, eig_list, wti,
                                                         temperature, ffval, eief = True, efixed = 14.7)
                        inner_vals2.append(val)
                    inner_vals.append(sum(inner_vals2))
                final_vals.append(inner_vals)
            final_vals = np.array(final_vals)
            
            en = clock()
            print en-st
    
            plot_cross_section(h_list,wt_list,final_vals)
            
            sys.exit()
    
        rad = 1.0
        tau = np.array([0,0,0])
        wt = np.array(1.0)
        
        x,y,z=run_spherical_averaging(N_atoms_uc,atom_list,csection,kapvect,tau_list,eig_list,wt_list,fflist,temperature)
        np.save(os.path.join(file_pathname,r'myfilex.txt'),x)
        np.save(os.path.join(file_pathname,r'myfiley.txt'),y)
        np.save(os.path.join(file_pathname,r'myfilez.txt'),z)
    
        EN = clock()
        print 'Total Run Time', EN - ST
    
        plot_cross_section(x,y,z,maxval=900)
        
    #    for h,k,l in zip(h_list,k_list,l_list):
    #        for ele in w_list:
    #            t = np.arccos(l)
    #            p = np.arctan2(k,h)
    #            eig = deepcopy(eig_list[0][0])
    #            kx = sp.Symbol('kx',real=True)
    #            ky = sp.Symbol('kx',real=True)
    #            kz = sp.Symbol('kx',real=True)
    #            S = sp.Symbol('S',real=True)
    #            eig = eig.subs(kx,h).subs(ky,k).subs(kz,l).subs(S,1.0).evalf(chop=True)
    #            rad = np.sqrt(h*h+k*k+l*l)
    #            res = spherical_averaging(N_atoms_uc, atom_list, rad, csection, tau, eig_list, ele, 0.0001,theta=t,phi=p)
    #            points.append(res)
    #    points = np.array(points)
    #    points = points.reshape((50,50)).T
    #    print points
        #print csdata
        #plot_cross_section(h_list,wtlist,points)
        #plot_cross_section(kapvect[:,0],wtlist,csdata)
