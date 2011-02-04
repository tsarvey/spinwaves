"""
Disclaimer
==========

This software was developed at the National Institute of Standards and Technology at the NIST Center for Neutron Research by employees of the Federal Government in the course of their official duties. Pursuant to title 17 section 105* of the United States Code this software is not subject to copyright protection and is in the public domain. The SPINAL software package is an experimental spinwave analysis system. NIST assumes no responsibility whatsoever for its use, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. The use of certain trade names or commercial products does not imply any endorsement of a particular product, nor does it imply that the named product is necessarily the best product for the stated purpose. We would appreciate acknowledgment if the software is used.

*Subject matter of copyright: United States Government works

Copyright protection under this title is not available for any work of the United States Government, but the United States Government is not precluded from receiving and holding copyrights transferred to it by assignment, bequest, or otherwise."""


from sympy import *
from sympy.printing import *

#-------------------------------------------------------------------------------

def subin(expression, pattern, replacement, match=True, *vars):
    # Might take out match optionality because will always need it.
    if match:
        check = lambda expression, pattern: expression.match(pattern)
    else:
        check = lambda expression, pattern: isinstance(expression,pattern)
    new = _walk_it(expression, pattern, replacement, check, vars)
    if new != None: return new
    else: return None

def _walk_it(expression, pattern, replacement, check, vars):
    op = expression.__class__
    if isinstance(type(expression),FunctionClass):
        new = [expression]; v = []
        if check(expression,pattern) != None:
            ch = list(check(expression,pattern).iteritems())
            for i in ch: v.append(i)
            new.insert(new.index(expression),replacement.subs(v))
            new.remove(expression)
        return Mul(*new)
    elif expression.args:
        new = [subexpression for subexpression in expression.args]; v = []
        for sub in new:
            if check(sub,pattern) != None:
                ch = list(check(sub,pattern).iteritems())
                for i in ch: v.append(i)
                new.insert(new.index(sub),replacement.subs(v))
                new.remove(sub)
            else: 
                new.insert(new.index(sub),_walk_it(sub, pattern, replacement, check, vars))
                new.remove(sub)
        return op(*new)
    else: return expression

#-------------------------------------------------------------------------------

def test_subin():
    a,b,c,d = symbols('abcd', commmutative = True)
    t,x,y,z = symbols('txyz', commutative = False)
    j = Wild('j'); k = Wild('k'); l = Wild('l');
    F = WildFunction('f')
    
    assert subin(a*exp(x*y), exp(j*k), DiracDelta(j-k)) == a*DiracDelta(x-y), 'Failed'; print '.'
    assert subin(a*exp(x*y) + b*exp(t*z), exp(j*k), cos(j-k)) == a*cos(x-y) + b*cos(t-z), 'Failed'; print '.'
    assert subin(a*exp(x*y*z - y*x*t)*cos(x), exp(j*z-k*t), DiracDelta(j-k)) == a*DiracDelta(x*y-y*x)*cos(x), 'a*exp(x*y*z - y*x*t)*cos(x) != a*DiracDelta(x*y-y*x)*cos(x)'; print '.'

#-------------------------------------------------------------------------------

if __name__ == '__main__':

    test_subin()



