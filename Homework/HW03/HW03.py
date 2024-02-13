# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 23:54:45 2024

@author: juliu
"""

import numpy as np
import matplotlib.pyplot as plt
import random

t = np.arange(0,np.pi+np.pi/60,np.pi/30)

y = np.cos(t)

def sum(n):
    for k in range(1,n+1):
        
        sum = 0
        sum = sum + t[k]*y[k]
    print("The sum is: ", sum)
        
    
sum(4)

theta = np.arange(0,2*np.pi + 1, 0.01)

x = 1.2*(1 + 0.1*np.sin(15*theta)*np.cos(theta))
y = 1.2*(1 + 0.1*np.sin(15*theta)*np.sin(theta))

#plt.plot(x,y)
#plt.show()

for i in range(1,11):
    p = random.uniform(0,2)
    xtheta = i * (1 + 0.05*np.sin((2+i)*theta+p))*np.cos(theta)
    ytheta = i * (1 + 0.05*np.sin((2+i)*theta+p))*np.sin(theta)
    
    plt.figure()
    plt.plot(xtheta,ytheta)
    plt.show()
    



