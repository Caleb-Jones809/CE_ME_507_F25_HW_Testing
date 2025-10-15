#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 06:48:53 2024

@author: kendrickshepherd
"""

import sys
import numpy as np
import matplotlib
#matplotlib.use("Qt5Agg")
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Given a set of points, pts, to interpolate
# and a polynomial degree, this function will
# evaluate the a-th basis function at the 
# location xi
def LagrangeBasisEvaluation(p,pts,xi,a):
    # ensure valid input
    if (p+1 != len(pts)):
        sys.exit("The number of input points for interpolating must be the same as one plus the polynomial degree for interpolating")

    # complete this code and return an appropriate value
    # as given in Hughes Equation 3.6.1
    reordered_pts = list(sorted(set(pts)))
    numerator = 1
    denominator = 1
    for b in range(len(reordered_pts)):
         # get numerator
        if b == a:
             pass
        else:
             numerator *= (xi - reordered_pts[b])
        # get denominator
        if b == a:
            pass
        else:
            denominator *= reordered_pts[a] - reordered_pts[b]
        
    return numerator/denominator


# Plot the Lagrange polynomial basis functions
# COMMENT THIS CODE
def PlotLagrangeBasisFunctions(p,pts,n_samples = 101):
    # create a list of n x's from smallest point to biggest point
    xis = np.linspace(min(pts),max(pts),n_samples)
    # create a figure to make the plot on
    fig, ax = plt.subplots()
    # run through the list just created and get the ath 
    # basis function evaluation at each point
    for a in range(0,p+1):
        vals = []
        for xi in xis:
            vals.append(LagrangeBasisEvaluation(p, pts, xi, a))
    # plot the points made with the ones calculated        
        plt.plot(xis,vals)

    ax.grid(linestyle='--')
    plt.savefig("Prob2_part3.png")



# Interpolate two-dimensional data
def InterpolateFunction(p,pts2D,n_samples = 101):
    # Insert appropriate text here, as described
    # in the homework prompt
    # seperate the lists
    pts = []
    coeffs = []
    for i in range(len(pts2D)):
        pts.append(pts2D[i][0])
        coeffs.append(pts2D[i][1])
    
    
    # double for loop to evaluate at the basis funtions
    xis = np.linspace(min(pts),max(pts),n_samples)
    ys = np.zeros_like(xis)
    for i in range(len(xis)):
        xi = xis[i]
        y = 0
        for j in range(p+1):
            y_val = LagrangeBasisEvaluation(p, pts, xi, j)
            y += coeffs[j] * y_val
        ys[i] = y

    # plot her
    fig, ax = plt.subplots()    
    plt.plot(xis,ys)
    plt.scatter(pts, coeffs,color='r')
    plt.savefig("Prob3.png")



    
mypts = np.linspace(-1,1,4)
myaltpts = [-1,0,.5,1]
p = 3

#PlotLagrangeBasisFunctions(p,mypts)

my2Dpts = [[0,1],[4,6],[6,2],[7,11]]
#InterpolateFunction(p,my2Dpts)

# problem 2
#PlotLagrangeBasisFunctions(1, [-1, 1])
#PlotLagrangeBasisFunctions(3, [-1, -1/3, 1/3, 1])
#PlotLagrangeBasisFunctions(3, [-1, 0, 1/2, 1])

# problem 3
pts2d = [[0,1], [4, 6], [6, 2], [7, 11]]
#InterpolateFunction(3, pts2d)
