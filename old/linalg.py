"""
linalg.py

Contains tools for working with linear-algebraic objects, in particular with
inner product spaces.
"""

import sympy
import itertools
import numpy as np
from np_utils import array_map

# Generating polynomial bases

# Convention: polynomials are represented as a tuple, where the first term
# is the basis element (as a sympy expression), and the second term is the
# total degree of the corresponding polynomial.

def _get_monomial_basis_tuples_1d(s, full_syms, deg_cap):
    """
    Returns the "reference" one-dimensional basis in variable s, up to 
    degree d. By default, just returns the powers s, s^2, ..., s^d.
    """
    return [(sympy.Poly(s**i, *full_syms, domain='RR'), i) for i in range(1, deg_cap+1)]

def _get_monomial_basis_tuples(syms, full_syms, deg_cap):
    if len(syms) == 0:
        return [(sympy.Poly(1, *full_syms, domain='RR'), 0)]
    if len(syms) == 1:
        return _get_monomial_basis_tuples_1d(syms[0], full_syms, deg_cap)

    b1 = _get_monomial_basis_tuples(syms[0:1], full_syms, deg_cap)
    b2 = _get_monomial_basis_tuples(syms[1:], full_syms, deg_cap)

    prod_basis = [
        (s*t, d+e)
        for ((s, d), (t, e)) in itertools.product(b1, b2)
        if d+e <= deg_cap
    ]

    return prod_basis

def get_monomial_basis(syms, full_syms, deg_cap):
    return [t[0] for t in _get_monomial_basis_tuples(syms, full_syms, deg_cap)]

def get_basis(space, ord_cap, deg_cap):
    syms = space.syms
    sample = space.sample
    
    # basis_dict has subsets of the symbol set as keys, and lists of basis
    # polynomials having those symbols in them as values.
    basis_dict = {
        () : [Polynomial(sympy.Poly(sympy.sympify('1'), *syms), space)]
    }

    for i in range(1, ord_cap+1):
        print "  order", i
        ord_basis_dict = {}

        for s in itertools.combinations(syms, i):
            print "---", s
            monomial_basis = get_monomial_basis(s, syms, deg_cap)
            ref_polys = [Polynomial(q, space) for q in monomial_basis]
            # print " - own", ref_polys
            
            lower_polynomials = []

            for j in range(len(s)):
                # get the subsets of syms having size j
                subsets = list(itertools.combinations(s, j))

                # get their basis elements
                for subset in subsets:
                    lower_polynomials += basis_dict[subset]

            # print " - lower", lower_polynomials

            orthogonalized_polys = []

            for p in ref_polys:
                q = Polynomial(
                    p.p - sum([p.poly_project_onto(q)
                             for q in lower_polynomials]
                    ), space
                )
                lower_polynomials.append(q)
                orthogonalized_polys.append(q)

            ord_basis_dict[s] = orthogonalized_polys
            
        basis_dict.update(ord_basis_dict)

    basis = sum(basis_dict.values(), [])

    tuples = basis_dict.keys()
    for s in tuples:        
        for t in tuples:
            if all(a in t for a in s) and len(t) > len(s):
                s_polys = basis_dict[s]
                t_polys = basis_dict[t]
                for p in s_polys:
                    for q in t_polys:
                        pass
                        # print p.dot(q)

    return basis

def eval_poly_list(polys, sample):
    # Returns a list of columns
    return [np.array([p.eval(tuple(x)) for x in sample]) for p in polys]

def is_subset_of(a, b):
    return all(t in b for t in a)

def get_basis_qr(space, ord_cap, deg_cap):
    full_syms = space.syms
    sample = space.sample

    N = sample.shape[0]
    d = sample.shape[1]

    # The basis is stored for the time being as a zipped list of polynomials
    # and the values that they take on the sample (storing the values so that
    # we do not need to recompute them each iteration).
    orthonormal_basis = {
        (): [
            (sympy.Poly(sympy.sympify(str(np.sqrt(1.0/N))), *full_syms),
             np.array([np.sqrt(1.0/N)]*N))
        ]
    }

    # TODO: normalizations for domain volume?
    
    for i in range(1, ord_cap+1):
        print "Order", i
        for syms in itertools.combinations(full_syms, i):
            # print "  - Variables", syms
            lower_polys, lower_poly_vals = zip(*sum([
                v
                for k, v in orthonormal_basis.items()
                if is_subset_of(k, syms)
            ], []))

            monomials = get_monomial_basis(syms, full_syms, deg_cap)
            monomial_vals = eval_poly_list(monomials, sample)

            A = np.column_stack(list(lower_poly_vals) + monomial_vals)
            Q, R = np.linalg.qr(A)
            
            # Get the part of Q that spans the monomials we added, this has
            # as its columns the values of our output polynomials, and also
            # the part of R that expresses the new basis in terms of the
            # monomials.
            tQ = Q[:,len(lower_polys):]
            # TODO: What if A has wrong shape?!
            tR = np.linalg.inv(R)[:,len(lower_polys):]

            all_polys = list(lower_polys) + monomials
            new_basis = [
                sum(c*m for m, c in zip(all_polys, coeffs))
                for coeffs in tR.T
            ]
            new_basis_zipped = zip(new_basis, tQ.T)

            orthonormal_basis[syms] = new_basis_zipped

    return orthonormal_basis

class Polynomial(object):
    def __init__(self, p, space):
        self.syms = space.syms
        self.p = p

        def f(x):
            return float(self.p.eval(tuple(x)))
        
        self.sample_values = array_map(f, space.sample)
        self.sq_norm = sum(self.sample_values**2)

    def dot(self, other):
        return np.sum(self.sample_values * other.sample_values)

    def poly_project_onto(self, other):
        return self.dot(other) / other.sq_norm * other.p
    
    def __call__(self, x):
        return self.p.eval(tuple(x))
    
    def __repr__(self):
        return repr(self.p)

class SampledPolynomialSpace(object):
    def __init__(self, sample, domain_volume):
        self.sample = sample

        self.d = sample.shape[1]
        self.N = sample.shape[0]
        self.V = domain_volume

        self.syms = [sympy.Symbol('x' + str(i+1)) for i in range(self.d)]

    def dot(self, e1, e2):
        e = sympy.Poly(e1 * e2, *self.syms)
        def integrand(x):
            substituted_expression = e.eval(
                tuple(x)
            )
            return float(substituted_expression)
        
        return np.sum(
            np.apply_along_axis(integrand, axis=1, arr=self.sample)
        ) * self.V / self.N

    def project_onto(self, e_from, e_onto):
        return self.dot(e_from, e_onto) / self.dot(e_onto, e_onto) * e_onto

    def get_projection_operator(self, e_onto):
        unit_vector = e_onto / self.dot(e_onto, e_onto)
        def f(e_from):
            return self.dot(e_from, e_onto) * unit_vector
        return f
