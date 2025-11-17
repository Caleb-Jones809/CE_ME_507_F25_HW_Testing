#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 11:37:32 2025

@author: kendrickshepherd
"""

import math
import numpy as np
import scipy
from scipy import sparse
from scipy import linalg

import sys
import itertools

import Gaussian_Quadrature_2D as gq
# import Gaussian_Quadrature_2D_Solution as gq

import unittest

class TestMultiDimensionalGaussianQuadrature(unittest.TestCase):

    def test_OneDQuadrature_Lagrange(self):
        n_quad = 3
        quadrature = gq.GaussQuadrature1D(n_quad)
        decimal_place = 3

        # Check the weights
        self.assertAlmostEqual(5./9., quadrature.quad_wts[0], decimal_place)
        self.assertAlmostEqual(8./9., quadrature.quad_wts[1], decimal_place)
        self.assertAlmostEqual(5./9., quadrature.quad_wts[2], decimal_place)

        # Check the points
        self.assertAlmostEqual(-np.sqrt(3./5.), quadrature.quad_pts[0], decimal_place)
        self.assertAlmostEqual(0, quadrature.quad_pts[1], decimal_place)
        self.assertAlmostEqual(np.sqrt(3./5.), quadrature.quad_pts[2], decimal_place)

        # Check the jacobian
        self.assertAlmostEqual(1., quadrature.jacobian, decimal_place)

    def test_OneDQuadrature_Bernstein(self):
        n_quad = 3
        quadrature = gq.GaussQuadrature1D(n_quad,start_pt=0,end_pt=1)
        decimal_place = 3

        # Check the weights
        self.assertAlmostEqual(5./9., quadrature.quad_wts[0], decimal_place)
        self.assertAlmostEqual(8./9., quadrature.quad_wts[1], decimal_place)
        self.assertAlmostEqual(5./9., quadrature.quad_wts[2], decimal_place)

        # Check the points
        self.assertAlmostEqual(-np.sqrt(3./5.)/2.+1./2., quadrature.quad_pts[0], decimal_place)
        self.assertAlmostEqual(1./2., quadrature.quad_pts[1], decimal_place)
        self.assertAlmostEqual(np.sqrt(3./5.)/2.+1./2., quadrature.quad_pts[2], decimal_place)

        # Check the jacobian
        self.assertAlmostEqual(1./2., quadrature.jacobian, decimal_place)

    def test_TwoDQuadrature_Lagrange(self):
        n_quad = 3
        quadrature = gq.GaussQuadratureQuadrilateral(n_quad)
        decimal_place = 3

        
        # Zip together points and values
        quad_wts = [val for val in quadrature.quad_wts]
        quad_pts = list(quadrature.quad_pts)
        quad_pts = [tuple(val) for val in quad_pts]
        paired = list(zip(quad_pts, quad_wts))

        # Sort by first, then second coordinate
        paired_sorted = sorted(paired, key=lambda x: (x[0][0], x[0][1]))
        
        # Unzip back into two lists
        points_sorted, weights_sorted = zip(*paired_sorted)
        
        # Convert back to lists if desired
        points_sorted = list(points_sorted)
        weights_sorted = list(weights_sorted)

        # Check the ordering of the weights
        self.assertAlmostEqual(5./9.*5./9., weights_sorted[0], decimal_place)
        self.assertAlmostEqual(8./9.*5./9., weights_sorted[1], decimal_place)
        self.assertAlmostEqual(5./9.*5./9., weights_sorted[2], decimal_place)
        self.assertAlmostEqual(5./9.*8./9., weights_sorted[3], decimal_place)
        self.assertAlmostEqual(8./9.*8./9., weights_sorted[4], decimal_place)
        self.assertAlmostEqual(5./9.*8./9., weights_sorted[5], decimal_place)
        self.assertAlmostEqual(5./9.*5./9., weights_sorted[6], decimal_place)
        self.assertAlmostEqual(8./9.*5./9., weights_sorted[7], decimal_place)
        self.assertAlmostEqual(5./9.*5./9., weights_sorted[8], decimal_place)

        # Check the ordering of the points
        # Sort by first, then second coordinate
        l = -np.sqrt(3./5.)
        c = 0
        r = np.sqrt(3./5.)

        self.assertAlmostEqual(l, points_sorted[0][0], decimal_place)
        self.assertAlmostEqual(l, points_sorted[1][0], decimal_place)
        self.assertAlmostEqual(l, points_sorted[2][0], decimal_place)
        self.assertAlmostEqual(c, points_sorted[3][0], decimal_place)
        self.assertAlmostEqual(c, points_sorted[4][0], decimal_place)
        self.assertAlmostEqual(c, points_sorted[5][0], decimal_place)
        self.assertAlmostEqual(r, points_sorted[6][0], decimal_place)
        self.assertAlmostEqual(r, points_sorted[7][0], decimal_place)
        self.assertAlmostEqual(r, points_sorted[8][0], decimal_place)

        self.assertAlmostEqual(l, points_sorted[0][1], decimal_place)
        self.assertAlmostEqual(c, points_sorted[1][1], decimal_place)
        self.assertAlmostEqual(r, points_sorted[2][1], decimal_place)
        self.assertAlmostEqual(l, points_sorted[3][1], decimal_place)
        self.assertAlmostEqual(c, points_sorted[4][1], decimal_place)
        self.assertAlmostEqual(r, points_sorted[5][1], decimal_place)
        self.assertAlmostEqual(l, points_sorted[6][1], decimal_place)
        self.assertAlmostEqual(c, points_sorted[7][1], decimal_place)
        self.assertAlmostEqual(r, points_sorted[8][1], decimal_place)


        # Check the jacobian
        self.assertAlmostEqual(1., quadrature.jacobian, decimal_place)

    # def test_TwoDQuadrature_Bernstein(self):
    #     n_quad = 3
    #     quadrature = gq.GaussQuadratureQuadrilateral(n_quad,start=0,end=1)
    #     decimal_place = 3

        
    #     # Zip together points and values
    #     quad_wts = [val for val in quadrature.quad_wts]
    #     quad_pts = list(quadrature.quad_pts)
    #     quad_pts = [tuple(val) for val in quad_pts]
    #     paired = list(zip(quad_pts, quad_wts))

    #     # Sort by first, then second coordinate
    #     paired_sorted = sorted(paired, key=lambda x: (x[0][0], x[0][1]))
        
    #     # Unzip back into two lists
    #     points_sorted, weights_sorted = zip(*paired_sorted)
        
    #     # Convert back to lists if desired
    #     points_sorted = list(points_sorted)
    #     weights_sorted = list(weights_sorted)

    #     # Check the ordering of the weights
    #     self.assertAlmostEqual(5./9.*5./9., weights_sorted[0], decimal_place)
    #     self.assertAlmostEqual(8./9.*5./9., weights_sorted[1], decimal_place)
    #     self.assertAlmostEqual(5./9.*5./9., weights_sorted[2], decimal_place)
    #     self.assertAlmostEqual(5./9.*8./9., weights_sorted[3], decimal_place)
    #     self.assertAlmostEqual(8./9.*8./9., weights_sorted[4], decimal_place)
    #     self.assertAlmostEqual(5./9.*8./9., weights_sorted[5], decimal_place)
    #     self.assertAlmostEqual(5./9.*5./9., weights_sorted[6], decimal_place)
    #     self.assertAlmostEqual(8./9.*5./9., weights_sorted[7], decimal_place)
    #     self.assertAlmostEqual(5./9.*5./9., weights_sorted[8], decimal_place)

    #     # Check the ordering of the points
    #     # Sort by first, then second coordinate
    #     l = -np.sqrt(3./5.)/2. + 0.5
    #     c = 0.5
    #     r = np.sqrt(3./5.)/2. + 0.5

    #     self.assertAlmostEqual(l, points_sorted[0][0], decimal_place)
    #     self.assertAlmostEqual(l, points_sorted[1][0], decimal_place)
    #     self.assertAlmostEqual(l, points_sorted[2][0], decimal_place)
    #     self.assertAlmostEqual(c, points_sorted[3][0], decimal_place)
    #     self.assertAlmostEqual(c, points_sorted[4][0], decimal_place)
    #     self.assertAlmostEqual(c, points_sorted[5][0], decimal_place)
    #     self.assertAlmostEqual(r, points_sorted[6][0], decimal_place)
    #     self.assertAlmostEqual(r, points_sorted[7][0], decimal_place)
    #     self.assertAlmostEqual(r, points_sorted[8][0], decimal_place)

    #     self.assertAlmostEqual(l, points_sorted[0][1], decimal_place)
    #     self.assertAlmostEqual(c, points_sorted[1][1], decimal_place)
    #     self.assertAlmostEqual(r, points_sorted[2][1], decimal_place)
    #     self.assertAlmostEqual(l, points_sorted[3][1], decimal_place)
    #     self.assertAlmostEqual(c, points_sorted[4][1], decimal_place)
    #     self.assertAlmostEqual(r, points_sorted[5][1], decimal_place)
    #     self.assertAlmostEqual(l, points_sorted[6][1], decimal_place)
    #     self.assertAlmostEqual(c, points_sorted[7][1], decimal_place)
    #     self.assertAlmostEqual(r, points_sorted[8][1], decimal_place)





if __name__ == '__main__':
    unittest.main()