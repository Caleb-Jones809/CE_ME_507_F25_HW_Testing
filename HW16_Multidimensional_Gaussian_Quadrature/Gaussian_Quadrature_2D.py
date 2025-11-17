#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 16:51:03 2023

@author: kendrickshepherd
"""

import math
import numpy as np
import scipy
from scipy import sparse
from scipy import linalg

import sys
import itertools

# Beta term from Trefethen, Bau Equation 37.6
def BetaTerm(n):
    if n <= 0:
        return 0
    else:
        return 0.5*math.pow((1-math.pow(2*n,-2)),-0.5)

# Theorem 37.4 from Trefethen, Bau
def ComputeQuadraturePtsWts(n):
    # Compute the Jacobi Matrix, T_n
    # given explicitly in Equation 37.6
    diag = np.zeros(n)
    off_diag = np.zeros(n-1)
    for i in range(0,n-1):
        off_diag[i] = BetaTerm(i+1)
        
    # Divide and conquer algorithm for tridiagonal
    # matrices
    # w is eigenvalues
    # v is matrix with columns corresponding eigenvectors
    [w,v] = scipy.linalg.eigh_tridiagonal(diag,off_diag,check_finite=False)
    
    # nodes of quadrature given as eigenvalues
    nodes = w
    # weights given as two times the square of the first 
    # index of each eigenvector
    weights = 2*(v[0,:]**2)
    
    return [nodes,weights]

class GaussQuadrature1D:
    
    def __init__(self,n_quad, start_pt = -1, end_pt = 1):
        self.n_quad = n_quad
        [self.quad_pts,self.quad_wts] = ComputeQuadraturePtsWts(self.n_quad)
        self.jacobian = 1
        
        if start_pt != -1 or end_pt != 1:
           self.__TransformToInterval__(start_pt,end_pt)
     
    def __TransformToInterval__(self,start,end):
        # complete this function
        new_pts = ((end - start)/2) * self.quad_pts + np.ones_like(self.quad_pts) * ((start+end) / 2)
        self.quad_pts = new_pts
        self.jacobian = (end-start) / 2
        

class GaussQuadratureQuadrilateral:
    
    def __init__(self,n_quad,start = -1,end = 1, deg=2):
        self.n_quad = n_quad
        self.degs = deg
        self.jacobian = 1
        [self.quad_pts,self.quad_wts] = ComputeQuadraturePtsWts(self.n_quad)
        self.start = start
        self.end = end
        if start != -1 or end != 1:
            self.__TransformToInterval__(start,end)

        
        # create the tensor product of quadrature points and weights
        # and Jacobians here and store them as quad_pts, quad_wts
        # and jacobian
        # initilaize
        new_quads = np.zeros((2, (len(self.quad_pts)**2)))
        new_quad_wts = np.zeros(len(self.quad_pts)**2)
        k = 0
        # get the appropriate pts
        for i in self.quad_pts:
            for j in self.quad_pts:
                new_quads[0, k] = i
                new_quads[1, k] = j
                k +=1
        # get wts
        k = 0
        for x in self.quad_wts:
            for y in self.quad_wts:
                new_quad_wts[k] = x*y
                k += 1
        self.quad_pts = new_quads
        self.quad_wts = new_quad_wts
        # get the jacobian
        new_jacob = ((end-start)/2) ** 2
        self.jacobian = new_jacob
        
    def __TransformToInterval__(self,start,end):
        new_pts = ((end - start)/2) * self.quad_pts + np.ones_like(self.quad_pts) * ((start+end) / 2)
        self.quad_pts = new_pts
        self.jacobian = (end-start) / 2


#object = GaussQuadratureQuadrilateral(3)
#print(object.quad_wts)
