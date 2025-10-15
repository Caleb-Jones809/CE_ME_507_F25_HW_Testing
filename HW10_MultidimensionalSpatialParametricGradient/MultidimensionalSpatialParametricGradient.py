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

from Univariate_Lagrange_Basis_Functions import LagrangeBasisEvaluation
from MultiDimensionalBasisFunctions import MultiDimensionalBasisFunction
from Univariate_Lagrange_Basis_Functions import LagrangeBasisEvaluation
from LagrangeBasisFuncDerivative import LagrangeBasisDervParamMultiD, LagrangeBasisParamDervEvaluation
#from MultiDimensionalJacobians import Nbasisfuncs EvalBasisFunction, EvalBasisDerivative, EvaluateFunctionParentDomain, EvaluateSpatialMapping, EvaluateDeformationGradient, EvaluateJacobian

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
        # IMPORT/COPY THIS FROM RECENT HOMEWORK
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

    # Evaluate the parametric gradient of a basis
    # function
    def EvaluateBasisParametricGradient(self,A, xi_vals):
        xi_derv = self.EvalBasisDerivative(A, xi_vals, dim=0)
        eta_derv = self.EvalBasisDerivative(A, xi_vals, dim=1)

        return [xi_derv, eta_derv]

    # Evaluate the parametric gradient of a basis
    # function
    def EvaluateBasisSpatialGradient(self,A, x_pts, xi_vals):
        # = J^-T *para_grad
        jacobian = self.EvaluateDeformationGradient(x_pts, xi_vals)
        inv_tran_jacob = np.linalg.inv(jacobian.T)
        para_grad = self.EvaluateBasisParametricGradient(A, xi_vals)
        return inv_tran_jacob @ para_grad
    
    def PlotJacobian(self,x_pts,npts=21,contours = False, parent_domain = False):
        return self.mdj.PlotJacobian(self,x_pts,npts=21,contours = False, parent_domain = False)

    # Grid plotting functionality that is used
    # in all other plotting functions
    def PlotGridData(self,X,Y,Z,npts=21,contours=False,xlabel=r"$x$",ylabel=r"$y$",zlabel=r"$z$", show_plot = True):
        if contours:
            fig, ax = plt.subplots()
            surf = ax.contourf(X,Y,Z,levels=100,cmap=matplotlib.cm.jet)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.colorbar(surf)
        else:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.jet,
                           linewidth=0, antialiased=False)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
        if show_plot:
            plt.show()
        
        return fig,ax

    # plot the mapping from parent domain to 
    # spatial domain            
    def PlotSpatialMapping(self,x_pts,npts=21,contours=False):
        dim = len(x_pts[0])
        
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)

        Xi,Eta = np.meshgrid(xivals,etavals)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)
        Z = np.zeros(Xi.shape)
        
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                X[j,i] = pt[0]
                Y[j,i] = pt[1]
                if dim == 3:
                    Z[i,j] = pt[2] 
        
        self.PlotGridData(X,Y,Z,contours=contours,)

    # plot a basis function defined on a parent
    # domain; this is similar to what was
    # in a previous homework, but slightly generalized                
    def PlotBasisFunctionParentDomain(self,A,npts=21,contours=False):
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        Z = np.zeros(Xi.shape)
        
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                Z[j,i] = self.EvalBasisFunction(A, xi_vals)

        self.PlotGridData(Xi,Eta,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$N(\xi,\eta)$")

    # plot a basis function defined on a spatial
    # domain
    def PlotBasisFunctionSpatialDomain(self,A,x_pts,npts=21,contours=False,on_parent_domain=True):
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        Z = np.zeros(Xi.shape)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)

        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                X[j,i] = pt[0]
                Y[j,i] = pt[1]
                Z[j,i] = self.EvalBasisFunction(A, xi_vals)
        
        self.PlotGridData(X,Y,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$N(\xi,\eta)$")


    # plot a solution field defined on a parent
    # domain
    def PlotParentSolutionField(self,d_coeffs,npts=21,contours = False):
        
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        Z = np.zeros(Xi.shape)
    
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                Z[j,i] = self.EvaluateFunctionParentDomain(d_coeffs,[xivals[i],etavals[j]])
    
        self.PlotGridData(Xi,Eta,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$u_h^e(\xi,\eta)$")

    # define a solution field mapped into the
    # spatial domain for an element
    def PlotSpatialSolutionField(self,d_coeffs,x_pts,npts=21,contours = False):
        
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
                X[j,i] = pt[0]
                Y[j,i] = pt[1]
                Z[j,i] = self.EvaluateFunctionParentDomain(d_coeffs,[xivals[i],etavals[j]])
    
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

    def PlotBasisFunctionGradient(self,A,x_pts,npts=21, parent_domain = True, parent_gradient = True):
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)
        Z = np.zeros(Xi.shape)
        U = np.zeros(Xi.shape)
        V = np.zeros(Xi.shape)
    
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                if not parent_domain:
                    pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                    X[j,i] = pt[0]
                    Y[j,i] = pt[1]
                if parent_gradient:
                    grad = self.EvaluateBasisParametricGradient(A, xi_vals)
                else:
                    grad = self.EvaluateBasisSpatialGradient(A, x_pts, xi_vals)
                U[j,i] = grad[0]
                V[j,i] = grad[1]
                Z[j,i] = self.EvalBasisFunction(A, xi_vals)

        if parent_domain:
            fig,ax = self.PlotGridData(Xi,Eta,Z,contours=True,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$J^e(\xi,\eta)$",show_plot = False)
            ax.quiver(Xi,Eta,U,V)
        else:
            fig,ax = self.PlotGridData(X,Y,Z,contours=True,zlabel=r"$J^e(x,y)$",show_plot = False)
            ax.quiver(X,Y,U,V)
        plt.savefig("hw10_4.png")



"""degx = 2
degy = 2
ipx = [-1, 0, 1]#, -1, 0, 1, -1, 0, 1]
ipy = [-1, 0, 1] #0, 0, 0, 1, 1, 1]
# first one
x_pts = [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2]]
object = LagrangeBasis2D(degx, degy, ipx, ipy)
object.PlotJacobian(x_pts, contours=True)"""



degx = 2
degy = 2
ipx = [-1, 0, 1]
ipy = [-1, 0, 1]
object = LagrangeBasis2D(degx, degy, ipx, ipy)
X_pts = [[0,0],[0,1],[1,1],[-1,0],[-1,2],[1,2],[-2,0],[-2,3],[1,3]]
A = 3
#object.PlotBasisFunctionGradient(A, X_pts, parent_domain=True, parent_gradient=True)
#object.PlotBasisFunctionGradient(A, X_pts, parent_domain=False, parent_gradient=True)
#object.PlotBasisFunctionGradient(A, X_pts, parent_domain=True, parent_gradient=False)
#object.PlotBasisFunctionGradient(A, X_pts, parent_domain=False, parent_gradient=False)