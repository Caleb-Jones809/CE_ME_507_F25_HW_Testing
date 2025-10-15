#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:18:02 2024

@author: kendrickshepherd
"""

import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt

# IMPORT (or copy) your code from HW3 here
# which evaluated a Lagrane polynomial basis function
import MultiDimensionalBasisFunctions
import Univariate_Lagrange_Basis_Functions

# x(\xi) = \sum_{a=0}^p x_a * N_a(\xi)
def XMap(deg,spatial_pts,interp_pts,xi):
    x_e = 0
    for a in range(deg+1):
# what to plug in for a?? are my plots what I am supposed to see??? what does the question at the end mean?
        x_e += spatial_pts[a]*Univariate_Lagrange_Basis_Functions.LagrangeBasisEvaluation(deg, interp_pts, xi, a)

    return x_e

def PlotXMap(deg,spatial_pts,interp_pts, npts=101, contours=True):
    # parametric points to evaluate in Lagrange basis function
    xi_vals = np.linspace(interp_pts[0],interp_pts[-1],npts)
    
    # evaluate and plot as a line
    if not contours:
        xs = []
        for i in range(0,len(xi_vals)):
            xi = xi_vals[i]
            xs.append(XMap(deg,spatial_pts,interp_pts,xi))
        
        plt.plot(xi_vals,xs)
        plt.savefig("hw6_task2_1")

    # evaluate as a contour plot
    else:
        Xi,Xi2 = np.meshgrid(xi_vals,[-0.2,0.2])
        Z = np.zeros(Xi.shape)
        for i in range(0,len(xi_vals)):
            xi = xi_vals[i]
            x = XMap(deg,spatial_pts,interp_pts,xi)
            for j in range(0,2):
                Z[j,i] = x

        fig, ax = plt.subplots()
        surf = ax.contourf(Z,Xi2,Xi,levels=100,cmap=matplotlib.cm.binary)
        ax.set_xlabel(r"$x$")
        ax.yaxis.set_ticklabels([])
        fig.colorbar(surf)
        plt.savefig("hw6_task2_1")



xa = [0, 1]
ca = [-1, 1]
#PlotXMap(1, xa, ca, contours=False)
xa2 = [.5, 1, 1.5]
ca2 = [-1, 0, 1]
#PlotXMap(2, xa2, ca2, contours=False)
xa3 = [.5, .7, 1.5]
ca3 = [-1, -.6, 1]
#PlotXMap(2, xa3, ca3, contours=False)
ca4 = [-1, 0, 1]
#PlotXMap(2, xa3, ca4, contours=False)