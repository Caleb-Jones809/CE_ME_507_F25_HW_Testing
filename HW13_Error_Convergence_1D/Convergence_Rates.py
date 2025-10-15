import numpy as np
from matplotlib import pyplot as plt

# Import appropriate functionality to complete this task
# here is some example text that could inform you how to do this

import Langrange_basis_functions as basis
import uniDimensionalXmap as xmap
import LagrangeBasisFuncDerivative as derv
import Gaussian_Quadrature as gq


def DeformationGradient1D(deg,x_bnds,xi):
    val = 0
    x_vals = np.linspace(x_bnds[0], x_bnds[-1], deg+1)
    xi_vals = np.linspace(-1,1,deg+1)
    for a in range(0,deg+1):
        val += x_vals[a] * derv.LagrangeBasisParamDervEvaluation(deg, xi_vals, xi, a)
    return val

# a is local basis function index
# e is element index
# output A which is global basis function index
def IEN(deg,a,e):
    return deg*e + a
    
# A is a basis function index with A >= 0
def ID(A, n_basis_functions):
    if A == n_basis_functions-1:
        return -1
    else:
        return A
    
    
def LocalForceMatrix(deg,x_bnds,f,gaussian):
    fe = np.zeros(deg+1)
    xi_pts = np.linspace(-1, 1, deg+1)
    x_vals = np.linspace(x_bnds[0], x_bnds[-1], deg+1)
    quad_wts = gaussian.quad_wts
    quad_pts = gaussian.quad_pts
    for g in range(0, gaussian.n_quad):
        xi = quad_pts[g]
        w = quad_wts[g]
        x_g = xmap.XMap(deg, x_vals, xi_pts, xi)
        jac = DeformationGradient1D(deg, x_bnds, xi)
        f_g = f(x_g)
        for a in range(0,deg+1):
            Na = basis.LagrangeBasisEvaluation(deg, xi_pts, xi, a)
            fe[a] += w * Na * f_g * jac
    
    return fe

def LocalForceBdryContrib(n_elems,e,h,g,ke):
    fe_bdry = np.zeros(ke.shape[0])
    if e == 0:
        fe_bdry[0] += h
    if e == n_elems-1:
        fe_bdry -= g*ke[:,-1]
    return fe_bdry

def LocalStiffnessMatrix(deg,x_bnds,gaussian):
    ke = np.zeros((deg+1,deg+1))
    xi_pts = np.linspace(-1, 1, deg+1)
    x_vals = np.linspace(x_bnds[0], x_bnds[-1], deg+1)
    quad_wts = gaussian.quad_wts
    quad_pts = gaussian.quad_pts
    for g in range(0, gaussian.n_quad):
        xi = quad_pts[g]
        w = quad_wts[g]
        jac = DeformationGradient1D(deg, x_bnds, xi)
        for a in range(0,deg+1):
            Na_x = derv.LagrangeBasisParamDervEvaluation(deg, xi_pts, xi, a)
            for b in range(0,deg+1):
                Nb_x = derv.LagrangeBasisParamDervEvaluation(deg, xi_pts, xi, b)
                ke[a,b] += w * (Na_x/jac) * (Nb_x/jac) * jac
    return ke
        

# solve the 1D finite element problem
# deg is polynomial degree
# x_nodes are a set of nodes 
def OneDFEM(deg, x_nodes, f, g, h, gaussian):
    
    n_elems = len(x_nodes)-1
    n_basis_funcs = deg * n_elems + 1
    n_unknowns = n_basis_funcs - 1
    
    F = np.zeros(n_unknowns)
    K = np.zeros((n_unknowns,n_unknowns))
    
    for e in range(0,n_elems):
        x_bnds = [x_nodes[e],x_nodes[e+1]]
        ke = LocalStiffnessMatrix(deg, x_bnds, gaussian)
        fe = LocalForceMatrix(deg, x_bnds, f, gaussian)
        fe_bdry = LocalForceBdryContrib(n_elems,e,h,g,ke)
        fe += fe_bdry
        for a in range(0,deg+1):
            A = IEN(deg,a,e)
            P = ID(A,n_basis_funcs)
            
            if P == -1:
                continue
            
            F[P] += fe[a]
            
            for b in range(0,deg+1):
                B = IEN(deg,b,e)
                Q = ID(B,n_basis_funcs)
                
                if Q == -1:
                    continue
                else:
                    K[P,Q] += ke[a,b]
                 
    # Right now we are not using h and g, which means we are
    # assuming that they are 0           
    d = np.linalg.solve(K,F)
    
    d_total = [val for val in d]
    d_total.append(g)
    
    return np.array(d_total)
    
def EvaluateSolution(e,deg,x_pts,xi_basis,xi,dtotal,deriv=0,derv_multiplier=1):
    uhval = 0
    for a in range(0,deg+1):
        A = IEN(deg,a,e)
        dA = dtotal[A]
        if deriv == 0:
            uhval += dA*basis.LagrangeBasisEvaluation(deg, xi_basis, xi, a)
        elif deriv == 1:
            J = DeformationGradient1D(deg,[x_pts[0],x_pts[-1]],xi)
            uhval += derv_multiplier*dA*derv.LagrangeBasisParamDervEvaluation(deg, xi_basis, xi, a) / J
        else:
            exit("Cannot compute on high-order Lagrange derivatives at this time")
    return uhval

def PlotSolutionCurve(deg,x_nodes,dtotal,n_samples = 5,deriv = 0, derv_multiplier = 1):
    uh = []
    x = []
    n_elems = len(x_nodes) - 1
    xi_basis = np.linspace(-1, 1, deg+1)
    xi_samples = np.linspace(-1,1,n_samples)
    for e in range(0,n_elems):
        x_pts = np.linspace(x_nodes[e],x_nodes[e+1],deg+1)
        for xi in xi_samples:
            xval = xmap.XMap(deg, x_pts, xi_basis, xi)
            uhval = EvaluateSolution(e,deg,x_pts,xi_basis,xi,dtotal,deriv,derv_multiplier)
            x.append(xval)
            uh.append(uhval)
    
    plt.plot(x,uh)


def ErrorNorm(exact_sol, D, num_elements, quad, deg, x_nodes, norm, exact_sol_derv):
    error = 0
    derv_error = 0
    pts = np.linspace(-1, 1, deg+1)
    # sum over elements
    for e in range(num_elements):
        # sum over quad points
        elem_e = 0
        for j in range(len(quad.quad_pts)):
            uh = 0
            # sum over degree
            for a in range(deg+1):
                # get uh(x(xi))
                index = IEN(deg, a, e)
                db = D[index]
                Nb = basis.LagrangeBasisEvaluation(deg, pts, quad.quad_pts[j], a)
                uh += db * Nb
            # get u(x(xi))
            u_x_xi = exact_sol(xmap.XMap(deg, x_nodes, pts, quad.quad_pts[j]))
            J = DeformationGradient1D(deg, (x_nodes[e], x_nodes[e+1]), quad.quad_pts[j])
            elem_e += (quad.quad_wts[j] * (u_x_xi - uh) ** 2) * J
        error += elem_e
    # add on derivitive part if H = 1
    if norm == 1:
        # (u,i - u,i^h) **2
        for e in range(num_elements):
        # sum over quad points
            elem_e_derv = 0
            for j in range(len(quad.quad_pts)):
                duh = 0
                J = DeformationGradient1D(deg, (x_nodes[e], x_nodes[e+1]), quad.quad_pts[j])
                # sum over degree
                for a in range(deg+1):
                    # get uh(x(xi))
                    index = IEN(deg, a, e)
                    db = D[index]
                    d_Nb = derv.LagrangeBasisParamDervEvaluation(deg,pts,quad.quad_pts[j],a)
                    duh += db * d_Nb * 1/J
                # get u(x(xi))
                du_x_xi = exact_sol_derv(xmap.XMap(deg, x_nodes, pts, quad.quad_pts[j]))
                elem_e_derv += (quad.quad_wts[j] * (du_x_xi - duh) ** 2) * 1/J
            derv_error += elem_e_derv
        return error + derv_error

    return error
    
def EvaluateConvergence(deg, num_intervals, L, f, g, h, gaussian, num_iterations, exact_sol, normtype=0, exact_sol_derv=lambda x:0):
    error_list = []
    h_list = []
    xi_basis = np.linspace(-1, 1, deg+1)
    
    if num_iterations < 2:
        exit("Cannot compute convergence rates on fewer than two data points")
    for i in range(0,num_iterations):
        # split mesh into 2^(number of spatial dimensions)^i
        # create a discretization with num_intervals equal elements
        interval = num_intervals*2**(i)
        x_nodes = np.linspace(0, L, interval)

        # compute u^h
        # solve for the solution vector, d
        D = OneDFEM(deg, x_nodes, f, g, h, gaussian)

        # compute error norm and store as list
        # compute the error in the solution from a given input solution
        error = ErrorNorm(exact_sol, D, interval-1, gaussian, deg, x_nodes, normtype, exact_sol_derv)
        error_list.append(error)

        # compoute h and store
        h = x_nodes[1] - x_nodes[0]
        h_list.append(h)
        # store the error and the element length in error_list and h_list, respectively
        

    # take the log of h_list and error_list values
    log_h_list = np.log(h_list)
    log_error_list = 2 * np.log(error_list)
    # find the slope of the log-h (in x) and log-error (in y) line
# still find slope
    plt.plot(log_h_list, log_error_list)
    plt.savefig("error_converg.png")

    m, b = np.polyfit(log_h_list, log_error_list, 1)
    print(m)
    return log_error_list
            
    
# Example software to use this code
def EvaluateAndPlotExample():
    deg = 1
    num_nodes = 10
    L = 1
    f = lambda x:np.exp(x)
    g = 1
    h = 1
    quadrature = gq.GaussQuadrature(deg)
    x_nodes = np.linspace(0,L,num_nodes)
    d = OneDFEM(deg, x_nodes, f, g, h, quadrature)
    print(d)
        
    PlotSolutionCurve(deg,x_nodes,d,n_samples=100,deriv=1)

#EvaluateAndPlotExample()


f = lambda x: np.exp(x)
g = 1
h = 1
deg = 5
num_intervals = 7
L = 1
gaussian = gq.GaussQuadrature(6)
num_iterations = 4
exact_sol_derv = f
norm = 1

EvaluateConvergence(deg, num_intervals, L, f, g, h, gaussian, num_iterations, exact_sol=f, exact_sol_derv = exact_sol_derv, normtype=norm)