"""
Tests for the CS-HDMR expansion.
"""

import random
import unittest
import sym_poly_utils
import sympy
import cs_hdmr
import numpy as np
import numpy.random
import tests.test_functions as test_functions
import model_validation
import functools
import logging

logging.basicConfig(level=logging.DEBUG)

random.seed()

class TestCSHDMR(unittest.TestCase):
    def setUp(self):
        self.modeler = cs_hdmr.cs_hdmr_modeler

    # Independently sampled polynomials
        
    def test_lin_poly_8(self):
        """
        Linear polynomial in 8 variables.
        """
        m = functools.partial(self.modeler, ord_cap=3, deg_cap=3)

        print model_validation.modeler_function_l2_errors(
            m, 80,
            model_validation.uniform_sampler(8, [-1.0, 1.0]),
            test_functions.lin_poly_8
        )

    # Correlated sampled polynomials
        
    def test_corr_poly_1(self):
        """
        Source: "General formulation of HDMR component functions with
        independent and correlated variables", Rabitz/Li, J Math Chem 2011.

        Two identical copies of the polynomial taken and added together (on
        distinct sets of variables) to increase dimensionality to make this
        a more interesting example.
        """
        mu = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        cov = np.array([
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.5, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        sample_fn = model_validation.multi_gaussian_sampler(mu, cov)

        test_fn = test_functions.gen_corr_poly_1(mu)

        m = functools.partial(self.modeler, ord_cap=3, deg_cap=4)
        print "Cross-validation relative L2 errors", model_validation.modeler_function_l2_errors(
            m, 200, sample_fn, test_fn
        )

if __name__ == '__main__':
    unittest.main()
