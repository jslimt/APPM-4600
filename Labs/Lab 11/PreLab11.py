# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:33:37 2024

@author: juliu
"""
import numpy as np

def composite_trapezoidal(a, b, f, N):

    h = (b-a)/N
    
    s = sum(f(a + (i * h)) for i in range(1, N))

    int_apprx = h*(0.5*(f(a)+f(b))+s)
    
    return int_apprx



