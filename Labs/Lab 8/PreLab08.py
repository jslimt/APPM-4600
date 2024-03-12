# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 21:29:02 2024

@author: juliu
"""

def f_eval(x0,y0,x1,y1,alpha):
    
    m = (y1-y0)/(x1-x0)
    
    f = lambda x: m*(x-x0)+y0
    
    print(f(alpha))
