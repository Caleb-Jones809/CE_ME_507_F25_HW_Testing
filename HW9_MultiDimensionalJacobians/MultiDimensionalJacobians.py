#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:06:03 2024

@author: kendrickshepherd
"""

import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt
from MultiDimensionalBasisFunctions import MultiDimensionalBasisFunction
from LagrangeBasisFuncDerivative import LagrangeBasisDervParamMultiD, LagrangeBasisParamDervEvaluation
from Univariate_Lagrange_Basis_Functions import LagrangeBasisEvaluation
# Copy or import functionality that you created 
# in previous homework assignments to complete
# this homework and minimize the amount of 
# work you have to repeat

# this class was created earlier in a previous
# assignment, but has been extended to cope with
# derivatives of basis functions and to plot
# Jacobians

# This is a class that describes a Lagrange basis
# in two dimensions
class LagrangeBasis2D:
    
    # initializor
    def __init__(self,degx,degy,interp_pts_x,interp_pts_y):
        self.degs = [degx,degy]
        self.interp_pts = [interp_pts_x,interp_pts_y]
        
    # the number of basis functions is the 
    # product of basis functions in the x (xi)
    # and y (eta) directions
    def NBasisFuncs(self):
        return (self.degs[0]+1) * (self.degs[1]+1)
        
    # basis function evaluation code from 
    # previous homework assignment
    # this should be imported from that assignment
    # or copied before this class is defined
    def EvalBasisFunction(self,A,xi_vals):
        return MultiDimensionalBasisFunction(A, self.degs, self.interp_pts, xi_vals)     
    
    # derivative of basis function code
    # from previous homework
    def EvalBasisDerivative(self,A,xis,dim):
        # IMPORT/COPY THIS FROM THE MOST RECENT HOMEWORK
        return LagrangeBasisDervParamMultiD(A, self.degs, self.interp_pts, xis, dim)


    # Evaluate a sum of basis functions times 
    # coefficients on the parent domain
    def EvaluateFunctionParentDomain(self, d_coeffs, xi_vals):
        u = 0
        nbfs = self.NBasisFuncs()
        for i in range(nbfs):
            u += d_coeffs[i] * self.EvalBasisFunction(i, xi_vals)

        return u
        
    # Evaluate the spatial mapping from xi and eta
    # into x and y coordinates
    def EvaluateSpatialMapping(self, x_pts, xi_vals):
        xe = 0  
        nbfs = self.NBasisFuncs()
        for i in range(nbfs): 
            xe += np.array(x_pts[i]) * self.EvalBasisFunction(i, xi_vals)
        return xe
    
    # Evaluate the Deformation Gradient (i.e.
    # the Jacobian matrix)
    def EvaluateDeformationGradient(self, x_pts, xi_vals):

        # initalize
        dxdxi, dxdeta = 0.0, 0.0
        dydxi, dydeta = 0.0, 0.0

        for a in range(self.NBasisFuncs()):
            # get derivatives
            xi_derv = self.EvalBasisDerivative(a, xi_vals, dim=0)
            eta_derv = self.EvalBasisDerivative(a, xi_vals, dim=1)

            # multiply by x_pts
            xa, ya = x_pts[a]
            dxdxi += xa * xi_derv
            dxdeta += xa * eta_derv
            dydxi += ya * xi_derv
            dydeta += ya*eta_derv

        # make DF
        DF = np.array([[dxdxi, dxdeta], [dydxi, dydeta]])
        return DF
    
    # Evaluate the jacobian (or the determinant
    # of the deformation gradient)
    def EvaluateJacobian(self, x_pts, xi_vals):
        # get DF
        DF = self.EvaluateDeformationGradient(x_pts, xi_vals)
        # return determinent
        return np.linalg.det(DF)
    
    # Grid plotting functionality that is used
    # in all other plotting functions
    def PlotGridData(self,X,Y,Z,npts=21,contours=False,xlabel=r"$x$",ylabel=r"$y$",zlabel=r"$z$"):
        if contours:
            fig, ax = plt.subplots()
            surf = ax.contourf(X,Y,Z,levels=100,cmap=matplotlib.cm.jet)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.colorbar(surf)
# RIGHT HERE
            plt.savefig("hw9_prob4_6.png")
            #plt.show()
        else:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.jet,
                           linewidth=0, antialiased=False)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
            plt.savefig("hw9_prob4_2.png")
            #plt.show()

            
    # plot the mapping from parent domain to 
    # spatial domain
    def PlotSpatialMapping(self,x_pts,npts=21,contours=False):
        dim = len(x_pts[0])
        
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)

        Xi,Eta = np.meshgrid(xivals,etavals)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)
        Z = np.zeros(Xi.shape)
        
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                X[i,j] = pt[0]
                Y[i,j] = pt[1]
                if dim == 3:
                    Z[i,j] = pt[2] 
        
        self.PlotGridData(X,Y,Z,contours=contours,)

    # plot a basis function defined on a parent
    # domain; this is similar to what was
    # in a previous homework, but slightly generalized
    def PlotBasisFunctionParentDomain(self,A,npts=21,contours=False):
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        Z = np.zeros(Xi.shape)
        
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                Z[i,j] = self.EvalBasisFunction(A, xi_vals)

        self.PlotGridData(Xi,Eta,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$N(\xi,\eta)$")

    # plot a basis function defined on a spatial
    # domain
    def PlotBasisFunctionSpatialDomain(self,A,x_pts,npts=21,contours=False,on_parent_domain=True):
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        Z = np.zeros(Xi.shape)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)

        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                X[i,j] = pt[0]
                Y[i,j] = pt[1]
                Z[i,j] = self.EvalBasisFunction(A, xi_vals)
        
        self.PlotGridData(X,Y,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$N(\xi,\eta)$")

    # plot a solution field defined on a parent
    # domain
    def PlotParentSolutionField(self,d_coeffs,npts=21,contours = False):
        
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        Z = np.zeros(Xi.shape)
    
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                Z[i,j] = self.EvaluateFunctionParentDomain(d_coeffs,[xivals[i],etavals[j]])
    
        self.PlotGridData(Xi,Eta,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$u_h^e(\xi,\eta)$")

    # define a solution field mapped into the
    # spatial domain for an element
    def PlotSpatialSolutionField(self,d_coeffs,x_pts,npts=21,contours = False):
        
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)
        Z = np.zeros(Xi.shape)
    
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                X[i,j] = pt[0]
                Y[i,j] = pt[1]
                Z[i,j] = self.EvaluateFunctionParentDomain(d_coeffs,[xivals[i],etavals[j]])
    
        self.PlotGridData(X,Y,Z,contours=contours,zlabel=r"$u_h^e(x,y)$")


    # plot Jacobians defined on the spatial 
    # or parent domain
    def PlotJacobian(self,x_pts,npts=21,contours = False, parent_domain = False):
        
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)
        Z = np.zeros(Xi.shape)
    
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                if not parent_domain:
                    pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                    X[j,i] = pt[0]
                    Y[j,i] = pt[1]
                Z[j,i] = self.EvaluateJacobian(x_pts,xi_vals)
    
        if parent_domain:
            self.PlotGridData(Xi,Eta,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$J^e(\xi,\eta)$")
        else:
            self.PlotGridData(X,Y,Z,contours=contours,zlabel=r"$J^e(x,y)$")




# prob 4
degx = 2
degy = 2
ipx = [-1, 0, 1]#, -1, 0, 1, -1, 0, 1]
ipy = [-1, 0, 1] #0, 0, 0, 1, 1, 1]
# first one
x_pts = [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2]]
object = LagrangeBasis2D(degx, degy, ipx, ipy)
#object.PlotJacobian(x_pts, contours=True)

# second one 
xpts2 = [[0,0],[0.5,0],[1,0],[0,1],[0.5,1],[1,1],[0,2],[0.5,2],[1,2]]
#object.PlotJacobian(xpts2, contours=True)

# third one
xpts3 = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
#object.PlotJacobian(xpts3, contours=True)

# fourth one
xpts4 = [[0,2],[0,1],[0,0],[1,2],[1,1],[1,0],[2,2],[2,1],[2,0]]
#object.PlotJacobian(xpts4, contours=True)

# fifth one
xpts5 = [[0,0],[1,0],[2,0],[1,1],[1,1],[1,1],[2,2],[1,2],[0,2]]
#object.PlotJacobian(xpts5, contours=True)

# sixth one
xpts6 = [[0,0],[0,1],[1,1],[-1,0],[-1,2],[1,2],[-2,0],[-2,3],[1,3]]
#object.PlotJacobian(xpts6, contours=True)
