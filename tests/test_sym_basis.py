"""
Tests for the symbolic QR hierarchically-orthogonal basis construction.
"""

import random
import unittest
import sym_poly_utils
import sympy
import numpy as np
import numpy.random

random.seed()

class TestQRBasis(unittest.TestCase):
    """
    Test of hierarchically-orthogonal basis construction using the iterated
    QR decomposition method.
    """    
    def setUp(self):
        self.d = 3
        self.N = 25

    def _validate_basis(self, sample, basis):
        self.assertTrue(
            sym_poly_utils.validate_sample_basis_ho(sample, basis)
        )
        self.assertTrue(
            sym_poly_utils.validate_sample_basis_n(sample, basis)
        )
        
    def _test_sample_basis_hon(self, row_generator):
        syms = [sympy.Symbol('x' + str(i+1)) for i in range(self.d)]
        
        sample = np.array(
            [row_generator() for i in range(self.N)]
        )
        basis = sym_poly_utils.sample_honb(sample, syms)
        
        self._validate_basis(sample, basis)

    # Tests of independent variables with various distributions
        
    def test_uniform_basis_hon(self):
        """
        Performs an HON basis test for uniform variables.
        """
        def row_generator():
            return [random.uniform(-1, 1) for i in range(self.d)]

        self._test_sample_basis_hon(row_generator)

    def test_gaussian_basis_hon(self):
        """
        Performs an HON basis test for Gaussian variables.
        """
        def row_generator():
            return [random.gauss(0, 1) for i in range(self.d)]

        self._test_sample_basis_hon(row_generator)

    def test_gamma_basis_hon(self):
        """
        Performs an HON basis test for Gamma-distributed variables.
        """
        def row_generator():
            return [random.betavariate(0.5, 0.5) for i in range(self.d)]

        self._test_sample_basis_hon(row_generator)

    # Tests of dependent variables with various distributions

    def test_multi_gaussian_basis_hon(self):
        def row_generator():
            return np.random.multivariate_normal(
                np.array([1.0, 2.0, 3.0]),
                np.array(
                    [[1.0, 2.0, 0.0],
                     [2.0, 1.0, 1.0],
                     [3.0, 0.0, 1.0]]
                )
            )
        
        self._test_sample_basis_hon(row_generator)
        
if __name__ == '__main__':
    unittest.main()
