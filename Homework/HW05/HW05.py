# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:04:52 2024

@author: juliu
"""
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm

def barycentric_lagrange_interpolation(x, f_x, z):
    n = len(x) - 1 
    m = len(z)      
    w = np.ones(n+1)
    
    for j in range(1, n+1):
        for k in range(j):
            w[k] /= (x[k] - x[j])
            w[j] *= (x[j] - x[k])
            
    p_z = np.zeros(m)
    
    for i in range(m):
        num = np.sum(w / (z[i] - x) * f_x)
        den = np.sum(w / (z[i] - x))
        p_z[i] = num/den

    return(p_z)


f = lambda x: 1/(1+(16*x)**2)

n = 32
h = 2/n
x = np.array([-1 + (i - 1) * h for i in range(1, n+2)])  

f_x = f(x)  
z = np.linspace(-1, 1, 100)  

#interpolated_values = barycentric_lagrange_interpolation(x, f_x, z)

#plt.plot(x, f_x, 'o', label='Data points')

x_highres = np.linspace(-1, 1, 1001)

#plt.plot(x_highres, f(x_highres), label='$f(x)$')
#plt.xlabel('x')
#plt.ylabel('f(x)')
#plt.title('Barycentric Lagrange Interpolation')
#plt.legend()
#plt.grid(True)
#plt.show()

xj = []
for j in range(0,n+1):
    xj.append(np.cos(((2*j+1)*np.pi)/(2*(n+1))))
    
#interpolated_values = barycentric_lagrange_interpolation(xj, f_x, z)
#plt.plot(xj,f_x,'o',label='Chebyshev Nodes')
#plt.plot(x_highres, f(x_highres), label='$f(x)')
#plt.xlabel('x')
#plt.ylabel('f(x)')
#plt.title('Lagrange w/ Chebyshev Nodes')
#plt.legend()
#plt.show()




def driver():
    
    f = lambda x: np.sin(9*x)
    a = 0
    b = 1
    
    
    ''' number of intervals'''
    Nint = 3
    xint = np.linspace(a,b,Nint+1)
    yint = f(xint)

    ''' create points you want to evaluate at'''
    Neval = 40
    xeval =  np.linspace(xint[0],xint[Nint],Neval+1)

    
    
    (M,C,D) = create_natural_spline(yint,xint,Nint)
    
    print('M =', M)
#    print('C =', C)
#    print('D=', D)
    
    yeval = eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D)
    
#    print('yeval = ', yeval)
    
    ''' evaluate f at the evaluation points'''
    fex = f(xeval)
        
    nerr = norm(fex-yeval)
    print('nerr = ', nerr)
    
    plt.figure()    
    plt.plot(xeval,fex,'ro-',label='exact function')
    plt.plot(xeval,yeval,'bs--',label='natural spline') 
    plt.legend
    plt.show()
     
    err = abs(yeval-fex)
    plt.figure() 
    plt.semilogy(xeval,err,'ro--',label='absolute error')
    plt.legend()
    plt.show()
    

def create_natural_spline(yint, xint, N):
    # create the right hand side for the linear system
    b = np.zeros(N + 1)
    # vector values
    h = np.zeros(N + 1)
    for i in range(1, N):
        hi = xint[i] - xint[i - 1]
        hip = xint[i + 1] - xint[i]
        b[i] = (yint[i + 1] - yint[i]) / hip - (yint[i] - yint[i - 1]) / hi
        h[i - 1] = hi
        h[i] = hip

    # create matrix so you can solve for the M values
    # This is made by filling one row at a time
    A = np.zeros((N + 1, N + 1))
    for j in range(1, N):
        A[j][j] = 2 * (h[j - 1] + h[j])
        A[j][j - 1] = h[j - 1]
        A[j][j + 1] = h[j]
    A[0, 0] = 1
    A[N, N] = 1

    Ainv = np.linalg.inv(A)
    M = np.dot(Ainv, b)

    # Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
        C[j] = yint[j] / h[j] - h[j] * M[j] / 6
        D[j] = yint[j + 1] / h[j] - h[j] * M[j + 1] / 6

    # Adjust for the last interval
    D[N - 1] = yint[N] / h[N - 1] - h[N - 1] * M[N] / 6

    return M, C, D

       
def eval_local_spline(xeval,xi,xip,Mi,Mip,C,D):
# Evaluates the local spline as defined in class
# xip = x_{i+1}; xi = x_i
# Mip = M_{i+1}; Mi = M_i

    hi = xip-xi
    yeval = ((Mip / (6 * hi)) * (xeval - xi)**3 +
             (Mi / (6 * hi)) * (xip - xeval)**3 +
             C * (xip - xeval) +
             D * (xeval - xi))
    return yeval 
    
    
def  eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):
    
    yeval = np.zeros(Neval+1)
    
    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp= xint[j+1]
        
#   find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])
#        print('yloc = ', yloc)
#   copy into yeval
        yeval[ind] = yloc

    return(yeval)
           
driver()                  


