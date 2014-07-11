"""
Implements a CS-HDMR modeler using the QR-decomposition method to construct
a basis on the sample, and using the Meridian CS method to determine the
sparse coefficients of the basis expansion.
"""

import numpy as np
import sym_poly_utils
import meridian_cs
import sympy
from collections import defaultdict
from generic_models import BasisExpansion

def cs_hdmr_expansion(inputs, outputs, basis_dict):
    print "Validating hierarchical orthogonality of basis:", sym_poly_utils.validate_sample_basis_ho(inputs, basis_dict)
    print "Validating normality of basis:", sym_poly_utils.validate_sample_basis_n(inputs, basis_dict)
    
    sensing_columns = []
    basis_polys = []
    basis_poly_syms = []
    for syms, poly_list in basis_dict.items():
        _, values = zip(*[(x.poly, x.vals) for x in poly_list])
        values = list(values)

        sensing_columns += values

        basis_polys += poly_list
        basis_poly_syms += [syms]*len(poly_list)

    # Construct sensing matrix
    # astype fixes speed issue
    A = np.column_stack(sensing_columns).astype(float)
    
    # coeffs = basis_pursuit.l1m(A, outputs, e=0)
    coeffs = list(meridian_cs.mbcs(A, outputs, 0.1))

    d = defaultdict(list)
    for syms, p, c in zip(basis_poly_syms, basis_polys, coeffs):
        d[syms] += [(p, c)]

    return BasisExpansion(d, inputs)

def cs_hdmr_modeler(inputs, outputs, ord_cap=3, deg_cap=3, independent=False):
    d = inputs.shape[1]
    syms = [sympy.Symbol('x' + str(i+1)) for i in range(d)]
    
    if independent:
        def get_range(i):
            in_i = np.array([x[i] for x in inputs])
            center = np.sum(in_i) / len(in_i)
            var = np.var(in_i)
            d = np.sqrt(12*var)
            s = 2*center
            b = (s + d) / 2
            a = s - b
            return [a, b]

        ranges = [get_range(i) for i in range(d)]

        basis_dict = sym_poly_utils.uniform_honb(
            ranges, inputs, syms, ord_cap=ord_cap, deg_cap=deg_cap
        )
    else:
        basis_dict = sym_poly_utils.sample_honb(
            inputs, syms, ord_cap=ord_cap, deg_cap=deg_cap
        )

    expansion = cs_hdmr_expansion(inputs, outputs, basis_dict)
    
    return expansion
