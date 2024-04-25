# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 00:04:47 2024

@author: juliu
"""

import math
import numpy as np
import scipy
from scipy import integrate, linalg, io, signal
from scipy.integrate import quad
import matplotlib.pyplot as plt

def trapezoidal_rule(a, b, n, f):
    h = (b-a)/n
    sum_f = 0.5*(f(a)+f(b))
    for k in range(1,n):
        sum_f += f(a+k*h)
    return h*sum_f

f= lambda s: 1/(1+s**2)


def composite_simpsons_rule(a, b, n, f):
    h = (b-a)/n
    x = np.linspace(a, b, n+1)
    y = f(x)
    int_aprx = h/3*(y[0]+y[n]+4*np.sum(y[1:n:2])+2*np.sum(y[2:n-1:2]))
    return int_aprx


def err_trap(a,b,n,f,tol,act):
    trap_apprx = trapezoidal_rule(a,b,n,f)
    err = act-trap_apprx
    if  abs(err) < tol:
        print('The error is within the tolerance and is',err)
    else:
        print('The error is not within the tolerance.')
        
def err_simps(a,b,n,f,tol,act):
    simp_apprx = composite_simpsons_rule(a, b, n, f)
    err = act- simp_apprx
    if abs(err)<tol:
        print('The error is within the tolerance and is',err)
    else:
        print('The error is not within the tolerance.')
        
t=10
G = lambda x:x**(t-1)*np.exp(-x)

T = trapezoidal_rule(0, 50, 200, G)
gamma= scipy.special.gamma(t)

print(T,gamma,abs(gamma-T))
