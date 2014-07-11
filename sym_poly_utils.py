"""
Utilities for constructing hierarchically orthonormal bases of polynomials
with respect to dot products over finite samples, as well as methods to test
the normality and hierarchical orthogonality of these bases.
"""

from __future__ import division
import sympy
import scipy.misc
import itertools
import numpy as np
from np_utils import array_map
import random
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

random.seed()

# Convention: in the private methods (starting with an underscore),
# polynomials are represented as a tuple, where the first term is the basis
# element (as a sympy polynomial expression), and the second term is the
# total degree of the corresponding polynomial.

def _get_monomial_basis_tuples_1d(s, full_syms, deg_cap):
    """
    Returns the "reference" one-dimensional basis in variable s, up to 
    degree d. By default, just returns the powers s, s^2, ..., s^d.
    """
    return [
        (sympy.Poly(s**i, *full_syms, domain='RR'), i)
        for i in range(1, deg_cap+1)
    ]

def _get_monomial_basis_tuples(syms, full_syms, deg_cap):
    """
    Returns the reference multi-dimensional product basis generated from
    products of the polynomials from the _get_monomial_basis_tuples_1d
    method above. By default, these are all monomials.
    """
    # Recurse: divide symbols into two subsets, get the bases for each one,
    # and multiply them together.

    # Base case
    if len(syms) == 0:
        return [(sympy.Poly(1, *full_syms, domain='RR'), 0)]
    if len(syms) == 1:
        return _get_monomial_basis_tuples_1d(syms[0], full_syms, deg_cap)

    # Otherwise, divide in two
    b1 = _get_monomial_basis_tuples(syms[0:1], full_syms, deg_cap)
    b2 = _get_monomial_basis_tuples(syms[1:], full_syms, deg_cap)

    # Construct and return the product basis
    return [
        (s*t, d+e)
        for ((s, d), (t, e)) in itertools.product(b1, b2)
        if d+e <= deg_cap
    ]

def get_monomial_basis(syms, full_syms, deg_cap):
    """
    Returns the same monomial basis as the above, but without the extra data
    of the polynomial degree in the second coordinate.
    """
    return [
        t[0] for t in _get_monomial_basis_tuples(syms, full_syms, deg_cap)
    ]

def eval_poly_list(sample, polys):
    """
    Returns a list of multivariate polynomials on a list of sample points,
    where the result is a list of "output vectors", the images of the sample
    under one polynomial at a time.
    """
    return [np.array([p.eval(tuple(x)) for x in sample]) for p in polys]

def _is_subset_of(a, b):
    """
    Returns whether a is a subset of b (utility for dealing with sets of
    symbols.
    """
    return all(t in b for t in a)

class SampledPolynomial:
    def __init__(self, poly, vals, inputs=None):
        self.poly = poly
        if vals is not None:
            self.vals = vals
        else:
            self.vals = np.array([poly.eval(tuple(x)) for x in inputs])

    def __str__(self):
        return str(self.poly)

    def __repr__(self):
        return str(self)
    
    def __add__(self, other):
        return SampledPolynomial(
            self.poly + other.poly,
            self.vals + other.vals
        )

    def __sub__(self, other):
        return SampledPolynomial(
            self.poly - other.poly,
            self.vals - other.vals
        )

    def __mul__(self, other):
        if isinstance(other, SampledPolynomial):
            return SampledPolynomial(
                self.poly * other.poly,
                self.vals * other.vals
            )
        else:
            # Assume other is a scalar
            return SampledPolynomial(
                self.poly * float(other),
                self.vals * float(other)
            )


    def __div__(self, other):
        return SampledPolynomial(
            self.poly / other.poly,
            self.vals / other.vals
        )

    def __call__(self, *args, **kwargs):
        return float(self.poly.eval(*args, **kwargs))
    
    def dot(self, other, normalize=False):
        ret = np.dot(self.vals, other.vals)
        if normalize:
            ret = ret / self.vals.shape[0]
        return ret
    
def sample_honb(sample, base_syms, ord_cap=3, deg_cap=3, method='qr'):
    """
    Returns a basis that is hierarchically orthonormal with respect to the
    finite inner product over the points of sample, on symbols base_syms.
    The basis is in the form of a dictionary, where keys are sets of symbols,
    and values are lists of basis polynomials (as SymPy Polynomial objects)
    non-constant with respect to precisely those symbols.
    """
    # Number of points
    N = sample.shape[0]
    # Dimension of ambient space
    d = sample.shape[1]

    logger.info("Constructing HON basis")
    logger.info(
        "N = " + str(N) + ", d = " + str(d) + \
            ", max order = " + str(ord_cap) + ", max degree = " + \
            str(deg_cap)
    )
    
    basis = {
        (): [
            (sympy.Poly(sympy.sympify(str(np.sqrt(1.0/N))), *base_syms),
             np.array([np.sqrt(1.0/N)]*N))
        ]   
    }

    for i in range(1, ord_cap+1):
        logger.debug("Constructing terms of order " + str(i))
        for syms in itertools.combinations(base_syms, i):
            # print syms
            lower_polys, lower_poly_vals = zip(*sum([
                v
                for k, v in basis.items()
                if _is_subset_of(k, syms)
            ], []))

            monomials = get_monomial_basis(syms, base_syms, deg_cap)
            monomial_vals = eval_poly_list(sample, monomials)

            A = np.column_stack(list(lower_poly_vals) + monomial_vals)
            # print A.shape
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

            basis[syms] = new_basis_zipped

    logger.info(
        "Constructed HON basis of " + \
            str(sum([len(v) for v in basis.values()])) + " polynomials"
    )

    def convert_sampled_poly_list(l):
        return [SampledPolynomial(p, v) for p, v in l]
    
    return {
        k: convert_sampled_poly_list(v)
        for k, v in basis.items()
    }

def uniform_honb(ranges, inputs, base_syms, ord_cap=3, deg_cap=3):
    def get_legendre_poly(n, sym, r):
        a = r[0]
        b = r[1]

        c = (a + b) / 2
        m = 2 / (b - a)

        print sym, n
        
        if n == 0:
            const_val = np.sqrt(1.0 / (b - a))
            return SampledPolynomial(
                sympy.Poly(const_val, *base_syms, domain='RR'),
                None, inputs=inputs
            )
        
        sym_n = (sym - c)*m
        
        p = sympy.expand((sym*sym - 1)**n)
        for i in range(n):
            p = p.diff()

        p = p.subs(sym, sym_n)
            
        return SampledPolynomial(
            sympy.Poly(
                (1/((2**n)*scipy.misc.factorial(n))) * np.sqrt(n + 0.5) * p,
                *base_syms, domain='RR'
            ),
            None, inputs=inputs
        )
        
    def get_legendre_basis_1d(deg_cap, sym, r):
        return [get_legendre_poly(i, sym, r) for i in range(deg_cap+1)]

    sym_bases = {
        s: get_legendre_basis_1d(deg_cap, s, r)
        for s, r in zip(base_syms, ranges)
    }

    print "A"

    basis_dict = {}
    basis_dict[()] = [np.prod([polys[0] for polys in sym_bases.values()])]
    for i in range(1, ord_cap+1):
        print i
        ix_ranges = [range(1, deg_cap+1) for j in range(i)]
        valid_indices = [
            t for t in itertools.product(*ix_ranges) if sum(t) <= deg_cap
        ]
        for syms in itertools.combinations(base_syms, i):
            print "   ", syms
            basis_polys = []
            for t in valid_indices:
                p = np.prod([sym_bases[syms[j]][t[j]] for j in range(i)])
                basis_polys.append(p)
            basis_dict[syms] = basis_polys
            
    return basis_dict
            
def sample_dot(sample, p1, p2, normalize=False):
    """
    Computes the dot product of polynomials p1 and p2 with respect to the
    points in sample.
    """
    if isinstance(p1, SampledPolynomial):
        ret = p1.dot(p2, normalize=normalize)
    else:
        y1, y2 = tuple(eval_poly_list(sample, [p1, p2]))
        ret = np.dot(y1, y2)

    if normalize:
       ret = ret / sample.shape[0] 

    return ret

def validate_sample_basis_n(sample, basis, tolerance=1.0e-08):
    """
    Validates that the basis elements are normal over the sample.
    """
    sym_sets = basis.keys()

    for s in sym_sets:
        s_polys = basis[s]

        # Test for normality
        for ps in s_polys:
            r = np.absolute(1.0 - sample_dot(sample, ps, ps, normalize=True))
            if r > tolerance:
                print r
                return False
             
    return True

def validate_sample_basis_ho(sample, basis, tolerance=1.0e-08):
    """
    Validates that the basis elements are hierarchically orthogonal over
    the sample.
    """
    sym_sets = basis.keys()

    for s in sym_sets:
        s_polys = basis[s]

        # Test for orthogonality        
        for t in sym_sets:
            if _is_subset_of(t, s) and t is not s:
                t_polys = basis[t]
                for ps in s_polys:
                    for pt in t_polys:
                        c = np.absolute(sample_dot(sample, ps, pt, normalize=True))
                        if c > tolerance:
                            print t, str(pt)
                            print s, str(ps)
                            print c
                            return False

    return True
