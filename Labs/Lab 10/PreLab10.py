# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 23:23:50 2024

@author: juliu
"""
import numpy as np
import matplotlib.pyplot as plt

def eval_legendre(n,x):
    p = [0] * n
    
    phi_0 = 1
    phi_1 = x

    i=0
    while i < n+1:
        
        p[0]=1
        p[1]=x
        p[i+1] = (1/(n+1))*((2*n+1)*x*p[i]-n*p[i-1])
        i += 1
        
    print(p)
    
    
    
    

    

        
        
        