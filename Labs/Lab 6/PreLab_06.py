# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 23:18:01 2024

@author: juliu
"""

import numpy as np

def fwrd_dif(f,s,h):
    f_prime = (f(s+h) - f(s))/h
    
    print(f_prime)
    
fwrd_dif(np.cos,np.pi/2,0.1)
fwrd_dif(np.cos,np.pi/2,0.00001)

def cntr_dif(f,s,h):
    f_prime = (f(s+h)-f(s-h))/(2*h)
    
    print(f_prime)
    
cntr_dif(np.cos,np.pi/2,0.00001)
