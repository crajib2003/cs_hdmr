"""
model_validation.py

Tools for model testing and validation. The type of modeling device used here is a "modeler", which takes an input matrix and output vector, and returns
a callable that performs the model approximation.
"""

import scripts.render_box_plots as bp
import matplotlib.pyplot as plt

import inspect
import logging
import random
import numpy as np
from np_utils import array_map
import operator
from collections import defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

random.seed()

# Sampling

def sample_to_io(sample):
    inputs = sample[:,0:-1]
    outputs = sample[:,-1]

    return inputs, outputs

def divide_sample(sample, num_divisions, shuffle=False):
    """
    Divides the passed sample (a NumPy array) into num_divisions subsamples
    sized as closely as possible to equally, randomizing if shuffle is True.
    """
    N = sample.shape[0]
    
    return [
        sample[ (N*k)/num_divisions : (N*(k+1))/num_divisions ]
        for k in range(num_divisions)
    ]

def make_gaussian_noise_fn(mu, sigma):
    def f():
        return random.gauss(mu, sigma)
    return f
    
# Samplers
    
def uniform_sampler(d, domain_1d):
    def f():
        return [
            random.uniform(*domain_1d) for i in range(d)
        ]
    return f

def multi_gaussian_sampler(mu, cov):
    def f():
        return np.random.multivariate_normal(mu, cov)
    return f
    
def make_function_sample(n, sample_fn, test_fn, noise=None):
    def get_row():
        inputs = sample_fn()
        output = test_fn(*inputs)
        if noise is not None:
            output += noise()
        # TODO: Why list() necessary here? Something with NumPy?
        return list(inputs) + [output]

    sample_list = [get_row() for i in range(n)]
    
    return np.array(sample_list)

# Testing modelers

def relative_l2_error(target, approx):
    error = np.linalg.norm(target - approx)
    print np.linalg.norm(target), np.linalg.norm(approx), error
    return error / np.linalg.norm(target)

def modeler_l2_errors(modeler, sample, sample_divisions=2):
    sample_list = divide_sample(sample, sample_divisions)
    
    l2_errors = []
    sens_dict = defaultdict(list)
    for i in range(len(sample_list)):
        training_sample = sample_list[i]
        training_in, training_out = sample_to_io(training_sample)
        model = modeler(training_in, training_out)

        sens_dict_i = model.sensitivity_dict(training_out)
        sorted_sens = sorted(
            sens_dict_i.iteritems(),
            key=lambda x: x[1]['t']
        )

        total = 0
        for syms, d in sorted_sens:
            sens_dict[syms].append(d['s'])
        
        # print "Component functions"
        # for s, p in model.component_functions(coeff_gate=1.0e-08).items():
        #     print "    ", s, p

        for j in range(len(sample_list)):
            if i != j:
                testing_sample = sample_list[j]
                test_in, test_out = sample_to_io(testing_sample)
                test_approx = array_map(model, test_in)
                l2_errors.append(
                    relative_l2_error(test_out, test_approx)
                )

    keys, vals = zip(*sorted(sens_dict.items(), key=lambda t: sum(t[1]), reverse=True))
    bp.plot_boxplots(vals[:15], [str(list(k)) for k in keys][:15])
    plt.show()
                                
    return l2_errors

def modeler_function_l2_errors(modeler, n, sample_fn, test_fn,
                               sample_divisions=2, noise=None):
    big_sample = make_function_sample(
        n, sample_fn, test_fn, noise=noise
    )

    ret = modeler_l2_errors(
        modeler, big_sample, sample_divisions=sample_divisions
    )

    logger.debug("Relative errors: " + str(ret))

    return ret
    
