# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:12:08 2024

@author: juliu
"""

import numpy as np
import matplotlib.pyplot as plt


def fixedpt(f,x0,tol,Nmax):
    ''' x0 = initial guess'''
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    count = 0
    ''' NEW SUBROUTINE '''
   
    approx_vec = [x0]
    while (count <Nmax):
        
        x1 = f(x0)
        approx_vec.append(x1)
        count = count +1
       
        if (abs(x1-x0) <tol):
                xstar = x1
                ier = 0
            
                #print("The approximations are:",approx_vec)
                return [xstar,ier], approx_vec
                
        x0 = x1
    
    xstar = x1
    ier = 1
    #print("The guesses were:",approx_vec)
    return [xstar, ier], approx_vec

f1 = lambda x: 0.5*np.sin(x)
output, approx = fixedpt(f1,.1,1e-5,100)
approx



def Aitkens(v):
    A_vec = []
      
    for n in range(len(v)-2):
        pn = v[n]- ((v[n+1]-v[n])**2)/(v[n+2]-2*v[n+1]+v[n])
        A_vec.append(pn)
    
    print(A_vec)
    
Aitkens(approx)

'''Writing the Subroutine'''


    