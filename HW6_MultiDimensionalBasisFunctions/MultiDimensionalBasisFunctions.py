#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:45:50 2024

@author: kendrickshepherd
"""

import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt

# IMPORT (or copy) your code from HW3 here
# which evaluated a Lagrane polynomial basis function
# import <UNCOMMENT THIS LINE AND INPUT YOUR FILENAME HERE>

# higher-dimensional basis function with multi-index
from Univariate_Lagrange_Basis_Functions import LagrangeBasisEvaluation
# return a basis function at an xi
def MultiDimensionalBasisFunctionIdxs(a, p, pts, xi):
    total = 1
    for i in range(len(a)):
        Na = LagrangeBasisEvaluation(p[i], pts[i], xi[i], a[i])
        total *= Na
    return total
    
# higher-dimensional basis function with single index
# return a basis function evaluated at an xi
def MultiDimensionalBasisFunction(A, p, pts, xi):
    a = []
    a.append(A % (p[0] + 1))
    for i in range(1, len(p)):
        diviser = 1
        for j in range(i):
            diviser *= (p[j] + 1)
        a.append(A // diviser)
    a = np.array(a)
    return MultiDimensionalBasisFunctionIdxs(a, p, pts, xi)
    
# plot of 2D basis functions with A a single index
def PlotTwoDimensionalParentBasisFunction(A,degs,npts = 101,contours = True):
    interp_pts = [np.linspace(-1,1,degs[i]+1) for i in range(0,len(degs))]
    xivals = np.linspace(-1,1,npts)
    etavals = np.linspace(-1,1,npts)
    
    Xi,Eta = np.meshgrid(xivals,etavals)
    Z = np.zeros(Xi.shape)
    
    for i in range(0,len(xivals)):
        for j in range(0,len(etavals)):
            Z[i,j] = MultiDimensionalBasisFunction(A, degs, interp_pts, [xivals[i],etavals[j]])
    
    # contour plot
    if contours:
        fig, ax = plt.subplots()
        surf = ax.contourf(Eta,Xi,Z,levels=100,cmap=matplotlib.cm.jet)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        fig.colorbar(surf)
        plt.savefig("hw6_task1_2.png")
    # 3D surface plot
    else:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(Eta, Xi, Z, cmap=matplotlib.cm.jet,
                       linewidth=0, antialiased=False)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        ax.set_zlabel(r"$N(\xi,\eta)$")
        plt.savefig("hw6_task1_2.png")


# the first
pt = np.linspace(-1, 1, 100)
p = [1, 1]
A = 1
#PlotTwoDimensionalParentBasisFunction(A, p, contours=False)
p2 = [2,1]
#PlotTwoDimensionalParentBasisFunction(A, p2, contours=False)
p3 = [2, 2]
#PlotTwoDimensionalParentBasisFunction(A, p3, contours=False)
p4 = [3, 3]
#PlotTwoDimensionalParentBasisFunction(A, p4, contours=False)