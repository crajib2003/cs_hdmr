"""
test_functions.py

Implements various test functions for creating benchmark datasets.
"""

import numpy as np

# Easy polynomial test cases

def lin_poly_3(x1, x2, x3):
    return x1 + x2 + x3

def lin_poly_8(x1, x2, x3, x4, x5, x6, x7, x8):
    return x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8

def semi_lin_poly_8(x1, x2, x3, x4, x5, x6, x7, x8):
    return x1*x2 + x3 + 0.5*x4*x5 + x6 + x7 + x8

def poly_3(x1, x2, x3):
    return x1 ** 4 + x2 ** 2 + x3 * x2

def poly_5(x1, x2, x3, x4, x5):
    return (x1**2)*(x2**2)*(x5) + (x2)*(x3) + (x4)*(x3**2)

def poly_8(x1, x2, x3, x4, x5, x6, x7, x8):
    return (x1**2)*(x2**2)*(x5) + (x2)*(x3) + (x4)*(x3**2) + x6*x7**3 + x8*x1

def gen_corr_poly_1(mu, a0=0, a1=1, b0=0, b1=1, c0=0, c1=1, c2=1, d0=0, d1=1, d2=1, d3=1):
    m1, m2, m3 = mu[0], mu[1], mu[2]

    def g1(x1, x2):
        return (a1*(x1 - m1) + a0)*(b1*(x2 - m2) + b0)

    def g2(x2):
        return c2*(x2 - m2)**2 + c1*(x2 - m2) + c0

    def g3(x3):
        return d3*(x3 - m3)**3 + d2*(x3 - m3)**2 + d1*(x3 - m3) + d0
    
    def f(x1, x2, x3):
        return g1(x1, x2) + g2(x2) + g3(x3)

    def h(x1, x2, x3, x4, x5, x6):
        return f(x1, x2, x3) + f(x4, x5, x6)
    
    return h

def ishigami(a1, a2, a3, a=7, b=0.1):
    """
    Example of highly input-sensitive function, used for benchmarking in
    sensitivity analysis.
    """
    x1 = np.pi * a1
    x2 = np.pi * a2
    x3 = np.pi * a3
    
    return np.sin(x1) + a*(np.sin(x2)**2) + b*(x3**4)*np.sin(x1)

def robot_arm(l1, l2, l3, l4, a1, a2, a3, a4):
    """
    Finding the effective length of a four-piece "robot arm", with specified
    piece lengths and the angles they make with each other.
    """
    t1 = np.pi * 2.0 * a1
    t2 = np.pi * 2.0 * a2
    t3 = np.pi * 2.0 * a3
    t4 = np.pi * 2.0 * a4

    u1 = l1 + np.cos(t1)
    u2 = l2 + np.cos(t1 + t2)
    u3 = l3 + np.cos(t1 + t2 + t3)
    u4 = l4 + np.cos(t1 + t2 + t3 + t4)
    u = u1 + u2 + u3 + u4

    v1 = l1 + np.sin(t1)
    v2 = l2 + np.sin(t1 + t2)
    v3 = l3 + np.sin(t1 + t2 + t3)
    v4 = l4 + np.sin(t1 + t2 + t3 + t4)
    v = v1 + v2 + v3 + v4

    return (u**2 + v**2)**(-0.5)

def dp_curved_function(x1, x2, x3):
    """
    Example of a highly curved function, possibly pathological case for Latin
    hypercube experimental design.
    
    Dette, H., & Pepelyshev, A. (2010).
    Generalized Latin hypercube design for computer experiments.
    Technometrics, 52(4).
    """
    return 4*(x1 - 2 + 8*x2 - 8*x2**2)**2 + (3 - 4*x2)**2 + \
        16*np.sqrt(x3 + 1)*(2*x3 - 1)**2
