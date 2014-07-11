"""
corr_hdmr.py
"""

import numpy as np
import linalg
import meridian_cs
import sympy
from collections import defaultdict
import operator

def cs_expansion_correlated(inputs, outputs, deg_cap=3, ord_cap=3):
    # Construct polynomial basis
    space = linalg.SampledPolynomialSpace(inputs, 2.0**inputs.shape[0])

    print "Constructing basis..."
    # basis = linalg.get_basis(space, ord_cap, deg_cap)
    basis_dict = linalg.get_basis_qr(space, ord_cap, deg_cap)

    sensing_columns = []
    basis_polys = []
    zipped_basis = []
    for syms, syms_zipped_basis in basis_dict.items():
        polys, values = zip(*syms_zipped_basis)
        polys = list(polys)
        values = list(values)
        sensing_columns += values
        zipped_basis += syms_zipped_basis
        basis_polys += [(p, syms) for p in polys]
    
    print "done.\n"

    # Construct sensing matrix
    print "Constructing sensing matrix..."
    A = np.column_stack(sensing_columns)

    print "Solving for compressed coefficients..."
    # coeffs = basis_pursuit.l1m(A, outputs, e=0)
    coeffs = list(meridian_cs.mbcs(A, outputs, 0.1))
    print "done.\n"
    
    return zip(zipped_basis, coeffs)

def basis_list_to_model(expansion):
    def f(x):
        return float(np.sum([c*p.eval(tuple(x)) for p, c in expansion]))
    return f

def cs_corr_modeler(inputs, outputs):
    zipped_basis = cs_expansion_correlated(inputs, outputs)
    d = defaultdict(list)
    for (p, vals), c in zipped_basis:
        atoms = p.as_expr().atoms(sympy.Symbol)
        k = tuple(sorted([str(a) for a in atoms]))
        d[k] += [(c**2)*(np.sum(vals**2))]

    print "Sensitivities:"
    sums = [(k, sum(v)) for k, v in d.items()]
    sorted_sums = sorted(sums, key=operator.itemgetter(1))
    for k, v in sorted_sums:
        print k, v
    
    return basis_list_to_model([(p, c) for (p, vals), c in zipped_basis])
