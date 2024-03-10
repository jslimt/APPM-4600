# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 23:04:33 2024

@author: juliu
"""

import numpy as np
from numpy import random as rand
import time
import math
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
import numpy as np


'''Problem 1'''

f = lambda x,y: 3*x**2 - y**2
g = lambda x,y: 3*x*y**2 - x**3 - 1

A = np.array([[1/16, 1/18], [0, 1/6]])
x0, y0 = 1, 1

num_iterations = 10
tol = 1e-6

x_values = [x0]
y_values = [y0]

for i in range(num_iterations):
    rhs = np.dot(A, np.array([[f(x_values[-1], y_values[-1])], [g(x_values[-1], y_values[-1])]]))
    
    x_new, y_new = np.array([[x_values[-1]], [y_values[-1]]]) - rhs
    
    x_values.append(x_new[0])
    y_values.append(y_new[0])
    
    if abs(x_values[-1] - x_values[-2]) < tol and abs(y_values[-1] - y_values[-2]) < tol:
        print(f"Converged after {i} iterations.")
        break

print(x_values,y_values)






'''Problem 2'''



def driver_2():
    ############################################################################
    ############################################################################
    # Rootfinding example start. You are given F(x)=0.

    #First, we define F(x) and its Jacobian.
    def F(x):
        return np.array([x[0]**2 + x[1]**2 - 4, np.exp(x[0]) + x[1] - 1]);

    def JF(x):
        return np.array([[2*x[0],2*x[1]],
                        [np.exp(x[0]),1]]);

    # Apply Newton Method:
    x0 = np.array([0,0]); tol=1e-6; nmax=100;
    (rN,rnN,nfN,nJN) = newton_method_nd(F,JF,x0,tol,nmax,True);
    print(rN)

    # Apply Lazy Newton (chord iteration)
    nmax=1000;
    (rLN,rnLN,nfLN,nJLN) = lazy_newton_method_nd(F,JF,x0,tol,nmax,True);

    # Apply Broyden Method
    Bmat='fwd'; B0 = JF(x0); nmax=100;
    (rB,rnB,nfB) = broyden_method_nd(F,B0,x0,tol,nmax,Bmat,True);

    # Plots and comparisons
    numN = rnN.shape[0];
    errN = np.max(np.abs(rnN[0:(numN-1)]-rN),1);
    plt.plot(np.arange(numN-1),np.log10(errN+1e-18),'b-o',label='Newton');
    plt.title('Newton iteration log10|r-rn|');
    plt.legend();
    plt.show();

    numB = rnB.shape[0];
    errB = np.max(np.abs(rnB[0:(numB-1)]-rN),1);
    plt.plot(np.arange(numN-1),np.log10(errN+1e-18),'b-o',label='Newton');
    plt.plot(np.arange(numB-1),np.log10(errB+1e-18),'g-o',label='Broyden');
    plt.title('Newton and Broyden iterations log10|r-rn|');
    plt.legend();
    plt.show();

    numLN = rnLN.shape[0];
    errLN = np.max(np.abs(rnLN[0:(numLN-1)]-rN),1);
    plt.plot(np.arange(numN-1),np.log10(errN+1e-18),'b-o',label='Newton');
    plt.plot(np.arange(numB-1),np.log10(errB+1e-18),'g-o',label='Broyden');
    plt.plot(np.arange(numLN-1),np.log10(errLN+1e-18),'r-o',label='Lazy Newton');
    plt.title('Newton, Broyden and Lazy Newton iterations log10|r-rn|');
    plt.legend();
    plt.show();


################################################################################
# Newton method in n dimensions implementation
def newton_method_nd(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1; nJ=0; #function and Jacobian evals
    npn=1;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        # compute n x n Jacobian matrix
        Jn = Jf(xn);
        nJ+=1;

        if verb:
            print("|--%d--|%1.7f|%1.7f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;

    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);

# Lazy Newton method (chord iteration) in n dimensions implementation
def lazy_newton_method_nd(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    # compute n x n Jacobian matrix (ONLY ONCE)
    Jn = Jf(xn);

    # Use pivoted LU factorization to solve systems for Jf. Makes lusolve O(n^2)
    lu, piv = lu_factor(Jn);

    n=0;
    nf=1; nJ=1; #function and Jacobian evals
    npn=1;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:

        if verb:
            print("|--%d--|%1.7f|%1.7f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -lu_solve((lu, piv), Fn); #We use lu solve instead of pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;

    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Lazy Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Lazy Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);

# Implementation of Broyden method. B0 can either be an approx of Jf(x0) (Bmat='fwd'),
# an approx of its inverse (Bmat='inv') or the identity (Bmat='Id')
def broyden_method_nd(f,B0,x0,tol,nmax,Bmat='Id',verb=False):

    # Initialize arrays and function value
    d = x0.shape[0];
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1;
    npn=1;

    #####################################################################
    # Create functions to apply B0 or its inverse
    if Bmat=='fwd':
        #B0 is an approximation of Jf(x0)
        # Use pivoted LU factorization to solve systems for B0. Makes lusolve O(n^2)
        lu, piv = lu_factor(B0);
        luT, pivT = lu_factor(B0.T);

        def Bapp(x): return lu_solve((lu, piv), x); #np.linalg.solve(B0,x);
        def BTapp(x): return lu_solve((luT, pivT), x) #np.linalg.solve(B0.T,x);
    elif Bmat=='inv':
        #B0 is an approximation of the inverse of Jf(x0)
        def Bapp(x): return B0 @ x;
        def BTapp(x): return B0.T @ x;
    else:
        Bmat='Id';
        #default is the identity
        def Bapp(x): return x;
        def BTapp(x): return x;
    ####################################################################
    # Define function that applies Bapp(x)+Un*Vn.T*x depending on inputs
    def Inapp(Bapp,Bmat,Un,Vn,x):
        rk=Un.shape[0];

        if Bmat=='Id':
            y=x;
        else:
            y=Bapp(x);

        if rk>0:
            y=y+Un.T@(Vn@x);

        return y;
    #####################################################################

    # Initialize low rank matrices Un and Vn
    Un = np.zeros((0,d)); Vn=Un;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        if verb:
            print("|--%d--|%1.7f|%1.7f|" % (n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        #Broyden step xn = xn -B_n\Fn
        dn = -Inapp(Bapp,Bmat,Un,Vn,Fn);
        # Update xn
        xn = xn + dn;
        npn=np.linalg.norm(dn);

        ###########################################################
        ###########################################################
        # Update In using only the previous I_n-1
        #(this is equivalent to the explicit update formula)
        Fn1 = f(xn);
        dFn = Fn1-Fn;
        nf+=1;
        I0rn = Inapp(Bapp,Bmat,Un,Vn,dFn); #In^{-1}*(Fn+1 - Fn)
        un = dn - I0rn;                    #un = dn - In^{-1}*dFn
        cn = dn.T @ (I0rn);                # We divide un by dn^T In^{-1}*dFn
        # The end goal is to add the rank 1 u*v' update as the next columns of
        # Vn and Un, as is done in, say, the eigendecomposition
        Vn = np.vstack((Vn,Inapp(BTapp,Bmat,Vn,Un,dn)));
        Un = np.vstack((Un,(1/cn)*un));

        n+=1;
        Fn=Fn1;
        rn = np.vstack((rn,xn));

    r=xn;

    if verb:
        if npn>tol:
            print("Broyden method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Broyden method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return(r,rn,nf)

# Execute driver




''' Problem 3 '''
'''
def driver_3_sdescent():
    ############################################################################
    ############################################################################
    # Rootfinding example start. You are given F(x)=0, and you find the minimizer
    # of q(x) = sum(F_j(x)^2).

    #First, we define F(x) and its Jacobian. These will help us find q and the
    #Gradient of q.
    def F(x):
        return np.array([[x[0]+np.cos(x[0]*x[1]*x[2])-1,
                         (1-x[0])**(-1/4)+x[1]+0.05*x[2]**2-0.15*x[2]-1,
                         -x[0]**2-0.01*x[1]**2+0.01*x[1]+x[2]-1]]);

    def JF(x):
        return np.array([[1-np.sin(x[0]*x[1]*x[2])*x[1]*x[2],-x[0]*x[2]*np.sin(x[0]*x[1]*x[2]),-x[0]*x[1]*np.sin(x[0]*x[1]*x[2])],
                        [(1/4)*(1-x[0])**(-3/4),1,0.1*x[2]**2-0.15],
                        [-2*x[0],-0.2*x[1]+0.01,1]]);

    # Define quadratic function and its gradient based on (F,JF)
    def q(x):
        Fun = F(x);
        return 0.5*(Fun[0]**2 + Fun[1]**2);

    def Gq(x):
        Jfun = JF(x);
        Ffun = F(x);
        return np.transpose(Jfun)@Ffun;

    # Apply steepest descent:
    x0=np.array([0,0,0]);
    tol=5*1e-2;
    nmax=1000;
    (r,rn,nf,ng)=steepest_descent(q,Gq,x0,tol,nmax);

    ################################################################################
    # plot of the trajectory of steepest descent against contour map
    nX=400;nY=400;
    (X,Y) = np.meshgrid(np.linspace(-2,10,nX),np.linspace(-2,8,nY));
    xx = X.flatten(); yy=Y.flatten();
    N = nX*nY;
    V = np.zeros((nX,nY));
    for i in np.arange(nX):
        for j in np.arange(nY):
            V[i,j]=q(np.array([X[i,j],Y[i,j]]));

    #levels=np.arange(0,200,1)
    fig=plt.contour(X,Y,V,levels=np.arange(0,200,1));

    plt.plot(rn[:,0],rn[:,1],'k-o');
    plt.show();

    ############################################################################
    ################################################################################
    # Plot of log||Fn|| and of log error
    Error = np.linalg.norm(np.abs(rn - r),axis=1);
    plt.plot(np.arange(rn.shape[0]),np.log10(Error),'r-o');
    plt.show();
    #input();

    Fn = np.zeros(len(rn))
    for i in np.arange(len(rn)):
        Fn[i] = q(rn[i]);

    plt.plot(np.arange(rn.shape[0]),np.log10(np.abs(Fn)),'g-o');
    plt.show();
    ################################################################################
    ################################################################################
    # Minimization example start. This is where you implement f and its gradient, and
    # use the steepest descent function above to find its minima given x0, tolerance
    # and max number of iterations nmax.

    # This example has a unique global minimizer at (1,1), with value equal to 0.
    # (Rosenbrock banana function)
    a=1; b=20;
    # objective function
    def fun(x):
        return (a - x[0])**2 + b*(x[1]-x[0]**2)**2;
    # gradient vector
    def Gfun(x):
        G = np.array([-2*(a-x[0])-4*x[0]*b*(x[1]-x[0]**2),2*b*(x[1]-x[0]**2)]);
        return G;
    # hessian matrix (2nd derivatives)
    def Hfun(x):
        H = np.array([[2-4*b*x[1]+12*b*x[0]**2,-4*b*x[0]],[-4*b*x[0],2*b]]);
        return H;

    ################################################################################
    # Apply steepest descent to finding the minima given initial conditions and tolerance
    x0=np.array([-1,-1]);
    tol=1e-6;
    nmax=1000;
    (r,rn,nf,ng)=steepest_descent(fun,Gfun,x0,tol,nmax);

    ################################################################################
    # plot of the trajectory of steepest descent against contour map
    nX=200;nY=200;
    (X,Y) = np.meshgrid(np.linspace(-1,1.5,nX),np.linspace(-1,1.5,nY));
    xx = X.flatten(); yy=Y.flatten();
    N = nX*nY;
    F = np.zeros((nX,nY));
    for i in np.arange(nX):
        for j in np.arange(nY):
            F[i,j]=fun(np.array([X[i,j],Y[i,j]]));

    fig=plt.contour(X,Y,F,levels=np.arange(0,20,0.25));

    plt.plot(rn[:,0],rn[:,1],'k-o');
    plt.show();

    ################################################################################
    # Plot of log||Fn|| and of log error
    Error = np.linalg.norm(np.abs(rn - np.array([1,1])),axis=1);
    plt.plot(np.arange(rn.shape[0]),np.log10(Error),'r-o');
    plt.show();
    #input();

    Fn = np.zeros(len(rn))
    for i in np.arange(len(rn)):
        Fn[i] = fun(rn[i]);

    plt.plot(np.arange(rn.shape[0]),np.log10(np.abs(Fn)),'g-o');
    plt.show();

################################################################################
# Backtracking line-search algorithm (to find an for the step xn + an*pn)
def line_search(f,Gf,x0,p,type,mxbck,c1,c2):
    alpha=2;
    n=0;
    cond=False; #condition (if True, we accept alpha)
    f0 = f(x0); # initial function value
    Gdotp = p.T @ Gf(x0); #initial directional derivative
    nf=1;ng=1; # number of function and grad evaluations

    # we backtrack until our conditions are met or we've halved alpha too much
    while n<=mxbck and (not cond):
        alpha=0.5*alpha;
        x1 = x0+alpha*p;
        # Armijo condition of sufficient descent. We draw a line and only accept
        # a step if our function value is under this line.
        Armijo = f(x1) <= f0 + c1*alpha*Gdotp;
        nf+=1;
        if type=='wolfe':
            #Wolfe (Armijo sufficient descent and simple curvature conditions)
            # that is, the slope at new point is lower
            Curvature = p.T @ Gf(x1) >= c2*Gdotp;
            # condition is sufficient descent AND slope reduction
            cond = Armijo and Curvature;
            ng+=1;
        elif type=='swolfe':
            #Symmetric Wolfe (Armijo and symmetric curvature)
            # that is, the slope at new point is lower in absolute value
            Curvature = np.abs(p.T @ Gf(x1)) <= c2*np.abs(Gdotp);
            # condition is sufficient descent AND symmetric slope reduction
            cond = Armijo and Curvature;
            ng+=1;
        else:
            # Default is Armijo only (sufficient descent)
            cond = Armijo;

        n+=1;

    return(x1,alpha,nf,ng);

################################################################################
# Steepest descent algorithm
def steepest_descent(f,Gf,x0,tol,nmax,type='swolfe',verb=True):
    # Set linesearch parameters
    c1=1e-3; c2=0.9; mxbck=10;
    # Initialize alpha, fn and pn
    alpha=1;
    xn = x0; #current iterate
    rn = x0; #list of iterates
    fn = f(xn); nf=1; #function eval
    pn = -Gf(xn); ng=1; #gradient eval

    # if verb is true, prints table of results
    if verb:
        print("|--n--|-alpha-|----|xn|----|---|f(xn)|---|---|Gf(xn)|---|");

    # while the size of the step is > tol and n less than nmax
    n=0;
    while n<=nmax and np.linalg.norm(pn)>tol:
        if verb:
            print("|--%d--|%1.5f|%1.7f|%1.7f|%1.7f|" %(n,alpha,np.linalg.norm(xn),np.abs(fn),np.linalg.norm(pn)));

        # Use line_search to determine a good alpha, and new step xn = xn + alpha*pn
        (xn,alpha,nfl,ngl)=line_search(f,Gf,xn,pn,type,mxbck,c1,c2);

        nf=nf+nfl; ng=ng+ngl; #update function and gradient eval counts
        fn = f(xn); #update function evaluation
        pn = -Gf(xn); # update gradient evaluation
        n+=1;
        rn=np.vstack((rn,xn)); #add xn to list of iterates

    r = xn; # approx root is last iterate

    return (r,rn,nf,ng);

################################################################################
if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver_3_sdescent()
'''

'''
def driver_3_newton():
    ############################################################################
    ############################################################################
    # Rootfinding example start. You are given F(x)=0.

    #First, we define F(x) and its Jacobian.
    def F(x):
        return np.array([[x[0]+np.cos(x[0]*x[1]*x[2])-1,
                         (1-x[0])**(-1/4)+x[1]+0.05*x[2]**2-0.15*x[2]-1,
                         -x[0]**2-0.01*x[1]**2+0.01*x[1]+x[2]-1]])

    def JF(x):
        return np.array([[1-np.sin(x[0]*x[1]*x[2])*x[1]*x[2],-x[0]*x[2]*np.sin(x[0]*x[1]*x[2]),-x[0]*x[1]*np.sin(x[0]*x[1]*x[2])],
                        [(1/4)*(1-x[0])**(-3/4),1,0.1*x[2]**2-0.15],
                        [-2*x[0],-0.2*x[1]+0.01,1]]);

    # Apply Newton Method:
    x0 = np.array([0,0,0]); tol=1e-6; nmax=100;
    (rN,rnN,nfN,nJN) = newton_method_nd(F,JF,x0,tol,nmax,True);
    print(rN)

    # Apply Lazy Newton (chord iteration)
    nmax=1000;
    (rLN,rnLN,nfLN,nJLN) = lazy_newton_method_nd(F,JF,x0,tol,nmax,True);

    # Apply Broyden Method
    Bmat='fwd'; B0 = JF(x0); nmax=100;
    (rB,rnB,nfB) = broyden_method_nd(F,B0,x0,tol,nmax,Bmat,True);

    # Plots and comparisons
    numN = rnN.shape[0];
    errN = np.max(np.abs(rnN[0:(numN-1)]-rN),1);
    plt.plot(np.arange(numN-1),np.log10(errN+1e-18),'b-o',label='Newton');
    plt.title('Newton iteration log10|r-rn|');
    plt.legend();
    plt.show();

    numB = rnB.shape[0];
    errB = np.max(np.abs(rnB[0:(numB-1)]-rN),1);
    plt.plot(np.arange(numN-1),np.log10(errN+1e-18),'b-o',label='Newton');
    plt.plot(np.arange(numB-1),np.log10(errB+1e-18),'g-o',label='Broyden');
    plt.title('Newton and Broyden iterations log10|r-rn|');
    plt.legend();
    plt.show();



    numLN = rnLN.shape[0];
    errLN = np.max(np.abs(rnLN[0:(numLN-1)]-rN),1);
    plt.plot(np.arange(numN-1),np.log10(errN+1e-18),'b-o',label='Newton');
    plt.plot(np.arange(numB-1),np.log10(errB+1e-18),'g-o',label='Broyden');
    plt.plot(np.arange(numLN-1),np.log10(errLN+1e-18),'r-o',label='Lazy Newton');
    plt.title('Newton, Broyden and Lazy Newton iterations log10|r-rn|');
    plt.legend();
    plt.show();


################################################################################
# Newton method in n dimensions implementation
def newton_method_nd(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1; nJ=0; #function and Jacobian evals
    npn=1;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        # compute n x n Jacobian matrix
        Jn = Jf(xn);
        nJ+=1;

        if verb:
            print("|--%d--|%1.7f|%1.7f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;

    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);

# Lazy Newton method (chord iteration) in n dimensions implementation
def lazy_newton_method_nd(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    # compute n x n Jacobian matrix (ONLY ONCE)
    Jn = Jf(xn);

    # Use pivoted LU factorization to solve systems for Jf. Makes lusolve O(n^2)
    lu, piv = lu_factor(Jn);

    n=0;
    nf=1; nJ=1; #function and Jacobian evals
    npn=1;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:

        if verb:
            print("|--%d--|%1.7f|%1.7f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -lu_solve((lu, piv), Fn); #We use lu solve instead of pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;

    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Lazy Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Lazy Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);

# Implementation of Broyden method. B0 can either be an approx of Jf(x0) (Bmat='fwd'),
# an approx of its inverse (Bmat='inv') or the identity (Bmat='Id')
def broyden_method_nd(f,B0,x0,tol,nmax,Bmat='Id',verb=False):

    # Initialize arrays and function value
    d = x0.shape[0];
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1;
    npn=1;

    #####################################################################
    # Create functions to apply B0 or its inverse
    if Bmat=='fwd':
        #B0 is an approximation of Jf(x0)
        # Use pivoted LU factorization to solve systems for B0. Makes lusolve O(n^2)
        lu, piv = lu_factor(B0);
        luT, pivT = lu_factor(B0.T);

        def Bapp(x): return lu_solve((lu, piv), x); #np.linalg.solve(B0,x);
        def BTapp(x): return lu_solve((luT, pivT), x) #np.linalg.solve(B0.T,x);
    elif Bmat=='inv':
        #B0 is an approximation of the inverse of Jf(x0)
        def Bapp(x): return B0 @ x;
        def BTapp(x): return B0.T @ x;
    else:
        Bmat='Id';
        #default is the identity
        def Bapp(x): return x;
        def BTapp(x): return x;
    ####################################################################
    # Define function that applies Bapp(x)+Un*Vn.T*x depending on inputs
    def Inapp(Bapp,Bmat,Un,Vn,x):
        rk=Un.shape[0];

        if Bmat=='Id':
            y=x;
        else:
            y=Bapp(x);

        if rk>0:
            y=y+Un.T@(Vn@x);

        return y;
    #####################################################################

    # Initialize low rank matrices Un and Vn
    Un = np.zeros((0,d)); Vn=Un;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        if verb:
            print("|--%d--|%1.7f|%1.7f|" % (n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        #Broyden step xn = xn -B_n\Fn
        dn = -Inapp(Bapp,Bmat,Un,Vn,Fn);
        # Update xn
        xn = xn + dn;
        npn=np.linalg.norm(dn);

        ###########################################################
        ###########################################################
        # Update In using only the previous I_n-1
        #(this is equivalent to the explicit update formula)
        Fn1 = f(xn);
        dFn = Fn1-Fn;
        nf+=1;
        I0rn = Inapp(Bapp,Bmat,Un,Vn,dFn); #In^{-1}*(Fn+1 - Fn)
        un = dn - I0rn;                    #un = dn - In^{-1}*dFn
        cn = dn.T @ (I0rn);                # We divide un by dn^T In^{-1}*dFn
        # The end goal is to add the rank 1 u*v' update as the next columns of
        # Vn and Un, as is done in, say, the eigendecomposition
        Vn = np.vstack((Vn,Inapp(BTapp,Bmat,Vn,Un,dn)));
        Un = np.vstack((Un,(1/cn)*un));

        n+=1;
        Fn=Fn1;
        rn = np.vstack((rn,xn));

    r=xn;

    if verb:
        if npn>tol:
            print("Broyden method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Broyden method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return(r,rn,nf)

# Execute driver

'''

'''Problem 4'''


