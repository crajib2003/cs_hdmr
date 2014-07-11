import scipy.special
import scipy.misc
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import random
from pyCSalgos.SL0.SL0 import SL0
from pyCSalgos.GAP.GAP import GAP
from sklearn import linear_model

import functools
import itertools
import operator
import csv

from collections import defaultdict

import basis_pursuit
import meridian_cs
import test_functions
from np_utils import array_map

# Some general utilities

def productify(f):
    """
    Turn a function f into a function that operates on each coordinate of
    a vector and returns the product of the results.
    """
    vectorized_f = np.vectorize(f)
    
    def productified_fun(x):
        return np.prod(vectorized_f(x))

    return productified_fun

def get_multi_indices(d, l, var_list=None):
    """
    Returns a list of all multi-indices with total degree d and length l,
    where if var_list is not None then it gives a list of indices that must
    be precisely the indices with non-zero value assigned.
    """
    def get_mi_full_vars(d, l, strictly_positive=False):
        if l == 1:
            if d == 0 and strictly_positive:
                return []
            else:
                return [[d]]
        else:
            if strictly_positive:
                sum_list = range(1, d)
            else:
                sum_list = range(d+1)
            return sum(
                [
                    [
                        [k] + i
                        for i in get_multi_indices(d-k, l-1)
                    ]
                    for k in sum_list
                ], []
            )

    if var_list is None:
        return get_mi_full_vars(d, l)
    else:
        var_values = get_mi_full_vars(
            d, len(var_list), strictly_positive=True
        )
        def expand_var_values(w):
            ret = [0] * l
            for i, v in enumerate(var_list):
                ret[v] = w[i]
            return ret
        return [expand_var_values(w) for w in var_values]

def get_multi_indices_up_to(d, l, var_list=None):
    """
    Returns a list of all multi-indices with total degree between 0 and d,
    and length l.
    """
    if var_list is not None and len(var_list) == 0:
        return [[0]*l]

    non_zero_mi = sum(
        [get_multi_indices(i+1, l, var_list=var_list) for i in range(d)], []
    )
    if var_list is not None:
        return non_zero_mi
    else:
        return [[0]*l] + non_zero_mi

def get_multi_indices_hdmr(d, l, order_limit):
    var_sets = sum(
        [
            list(itertools.combinations(range(l), r))
            for r in range(order_limit+1)
        ],
        []
    )

    return sum(
        [get_multi_indices_up_to(d, l, var_list=v) for v in var_sets],
        []
    )
        
# Orthogonal basis things

class OrthogonalProductBasis:
    """
    A product orthogonal function basis arising from some one-dimensional
    orthogonal basis on a domain of real numbers.
    """
    
    def __init__(self, weight, sq_norm, basis_eval, domain_volume):
        self.weight = productify(weight)
        self.sq_norm = productify(sq_norm)
        self.eval_1d = basis_eval
        def tensor_basis_eval(n, x):
            return np.prod(
                [basis_eval(*t) for t in zip(n, x)]
            )
        self.basis_eval = tensor_basis_eval
        self.domain_volume = domain_volume

    def __call__(self, n, x):
        return np.prod(
            [self.eval_1d(i, y) for i, y in zip(n, x)]
        )
    
    def coeff(self, n, inputs, outputs):
        def get_integrand(x):
            return self.weight(x) * self.basis_eval(n, x)
    
        input_dimension = inputs.shape[1]

        int_vals = np.apply_along_axis(
            get_integrand, axis=1, arr=inputs
        ) * outputs

        int_approx = np.sum(int_vals) / len(int_vals) \
            * (self.domain_volume**input_dimension)

        print int_approx / self.sq_norm(n)
        print "---\n"
        assert(False)
        return int_approx / self.sq_norm(n)

class BasisExpansion:
    def __init__(self, basis, expansion):
        """
        The constructor takes the polynomial basis in which the expansion
        is taken, and the expansion (a zipped list of basis element indices
        and coefficients).
        """
        self.basis = basis
        self.expansion = expansion

    def __call__(self, x):
        return np.sum([c * self.basis(n, x) for n, c in self.expansion])
    
    def hdmr_component_norms(self):
        d = defaultdict(list)

        for n, c in self.expansion:
            key = tuple([index for index, i in enumerate(n) if i != 0])
            d[key].append(self.basis.sq_norm(n) * (c**2))

        return {
            k: sum(v)
            for k, v in d.items()
        }
    
def monte_carlo_expansion(basis, inputs, outputs, degree_limit=6):
    input_dimension = inputs.shape[1]
    multi_indices = get_multi_indices_up_to(
        degree_limit, input_dimension
    )
    print "Total coefficients:", len(multi_indices)
    return BasisExpansion(
        basis, [
            (i, basis.coeff(i, inputs, outputs))
            for i in multi_indices
        ]
    )
            
def cs_expansion(basis, inputs, outputs, degree_limit=5, order_limit=3):
    input_dimension = inputs.shape[1]
    multi_indices = get_multi_indices_hdmr(
        degree_limit, input_dimension, order_limit=order_limit
    )

    # Dimensions of the sensing matrix
    M = len(multi_indices)
    print "M =", M
    N = inputs.shape[0]
    print "N =", N

    # Function generating the sensing matrix coefficients (i.e. the basis
    # functions evaluated on the input vectors)
    def matrix_entry(i, j):
        index = multi_indices[j]
        sample = inputs[i]
        return basis(index, sample)
    
    # Sensing matrix isan N x M matrix, which we right-multiply with the
    # vector of coefficients in the optimization.
    # TODO: faster way?
    A = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            A[i, j] = matrix_entry(i, j)

    print "- Running L1 minimization..."
    # coeffs = basis_pursuit.l1m(A, outputs, e=0)
    # coeffs = SL0(A, outputs, 0)
    params = {
        "num_iteration" : 1000,
        "greedy_level" : 0.9,
        "stopping_coefficient_size" : 1e-4,
        "l2solver" : 'pseudoinverse',
        "noise_level": 0
    }
    # coeffs = GAP(outputs, A, A.T, np.identity(M), np.identity(M), params,
    #            np.zeros(M))[0]
    clf = linear_model.Lasso(alpha = 0.001)
    clf.fit(A, outputs)
    coeffs = clf.coef_
    print "- L1 minimization done"
    
    return BasisExpansion(basis, zip(multi_indices, coeffs))

# Example: the basis of Chebyshev T-polynomials

def cheb_weight(x):
    return (1 - x**2)**(-0.5)

def cheb_sq_norm(n):
    if n == 0:
        return np.pi
    else:
        return np.pi / 2

chebyt_product_basis = OrthogonalProductBasis(
    cheb_weight,
    cheb_sq_norm,
    sp.special.eval_chebyt,
    2.0
)

leg_product_basis = OrthogonalProductBasis(
    lambda x: 1,
    lambda n: 2.0 / (2.0*n + 1.0),
    sp.special.eval_legendre,
    2.0
)

# HDMR

def rs_hdmr(sample):
    """
    :param sample: the data sample to calculate with
    :param order: the maximum order of expansion terms to compute
    :param poly_deg: the degree of Chebyshev polynomial approximation to use
    """
    inputs = sample[:,0:-1]
    outputs = sample[:,-1]

    exp = cs_expansion(
        chebyt_product_basis, inputs, outputs, degree_limit=3
    )

    exp_outputs = np.apply_along_axis(
        exp, axis=1, arr=inputs
    )
    
    l2_error = np.linalg.norm(outputs - exp_outputs, ord=2)
    print (outputs - exp_outputs)
    print l2_error / np.linalg.norm(outputs)
    
    cs_exp = cs_expansion(
        chebyt_product_basis, inputs, outputs
    )
    
    hdmr_weights = mc.hdmr_component_norms()
    hdmr_weights2 = cs.hdmr_component_norms()
    
    sorted_weights = sorted(
        hdmr_weights.iteritems(),
        key=operator.itemgetter(1),
        reverse=True
    )

    sorted_weights2 = sorted(
        hdmr_weights2.iteritems(),
        key=operator.itemgetter(1),
        reverse=True
    )
    
    for t, w in sorted_weights:
        print t, "---", w

    print "-------------------------"
        
    for t, w in sorted_weights2:
        print t, "---", w

    integrand_vals = np.apply_along_axis(
        cheb_weight, axis=1, arr=sample_inputs
    ) * (sample_outputs ** 2)
        
    print "L2 norm of sample", np.sum(integrand_vals) / len(integrand_vals) * (2**3)

def make_sample(
    num_samples, bounding_box, row_function, num_function_inputs, domain_1d,
    randomize=False
):
    """
    Creates a dataset where the rows represent applications of row_function
    at a random point. The number of samples used is num_samples, the
    argument bounding_box is a list of the domain intervals for each
    variable, and domain_1d is the domain to scale the inputs into (i.e.
    the domain of the orthogonal polynoimals in each variable).
    """
    
    def get_row():
        inputs = [
            random.uniform(*domain_1d) for i in range(num_function_inputs)
        ]
        output = row_function(*inputs)
        if randomize:
            output *= (1 + random.random()/10.)
        return inputs + [output]

    sample_list = [get_row() for i in range(num_samples)]

    return np.array(sample_list)

def run_training_test(sample, expander, training_proportion=0.5):
    training_data, test_data = divide_sample(sample, training_proportion)

    training_in, training_out = sample_to_io(training_data)
    test_in, test_out = sample_to_io(test_data)

    trained_expansion = expander(training_in, training_out)

    test_approx = array_map(trained_expansion, test_in)
    training_approx = array_map(trained_expansion, training_in)
    
    print "Training L2 error:", \
        relative_l2_error(training_out, training_approx)
    print "Training L1 error:", \
        average_relative_l1_error(training_out, training_approx)
    print "Test L1 error:", \
        average_relative_l1_error(test_out, test_approx)
    
    return relative_l2_error(test_out, test_approx)

def run_cs_test(sample, basis, trials=1):
    expander = functools.partial(cs_expansion, basis)

    if trials == 1:
        return run_training_test(sample, expander)
    
    results = []
    for i in range(trials):
        np.random.shuffle(sample)
        results.append(run_training_test(sample, expander))

    return results    

def run_cs_function_test(function, num_args):
    random.seed()

    errors = []
    for i in range(1):
        # print i
        sample = make_sample(
            120, [-1.0, 1.0], function, num_args, [-1.0, 1.0]
        )
        # e = run_cs_test(sample, leg_product_basis)
        # print "Test L2 error:", e
        # errors.append(e)

        e = run_cs_correlated_test(sample)
        print e

    # hist, bins = np.histogram(errors, bins=20)
    # width = 0.7 * (bins[1] - bins[0])
    # center = (bins[:-1]  + bins[1:]) / 2
    # plt.bar(center, hist, align='center', width=width)
    # plt.show()

def parkinsons_updrs_test():
    inputs = []
    outputs_1 = []
    outputs_2 = []
    with open('parkinsons_updrs.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        reader.next()
        for row in reader:
            outputs_1.append(float(row[4]))
            outputs_2.append(float(row[5]))
            inputs.append([float(r) for r in row[1:2] + row[3:4] + row[6:]])

    def io_merge(inputs, outputs):
        l = [i + [o] for i, o in zip(inputs, outputs)]
        return np.array(l)

    sample = io_merge(inputs, outputs_1)[:200]
    print run_cs_test(sample, chebyt_product_basis)
            
def main():
    run_cs_function_test(test_functions.lin_poly_3, 3)
    # parkinsons_updrs_test()
    
if __name__ == '__main__':
    main()
