"""
meridian_cs.py

Implements Bayesian compressed sensing using meridian priors.
(Bayesian Compressed Sensing Using Generalized Cauchy Priors, Carrillo et al)
"""

import numpy as np
import numpy.linalg
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def LLp_norm(x, p, delta):
    return np.sum(np.log(1 + delta**(-p) * np.abs(x)**p))

def LL1_norm(x, delta):
    return np.sum(np.log(1 + delta*np.abs(x)))

def W_inv(x, delta):
    x_a = np.abs(x)
    return np.diag((delta + x_a)*x_a)

def matrix_prod(*l):
    return reduce(np.dot, l)

def mbcs(phi, y, noise_variance, min_delta=1.0e-8, min_variation=0.000000005,
         max_iterations=1000):
    m, n = phi.shape
    logger.info("Sensing matrix dimensions " + str(phi.shape))
    assert(m < n)
    
    phi_t = np.transpose(phi)
    # lambda * identity matrix
    lId = 2.0 * noise_variance**2 * np.identity(m)
    
    iteration = 0

    M = np.dot(phi, phi_t)
    x = matrix_prod(
        phi_t,
        np.linalg.inv(M + lId),
        y
    )

    x_prev = None

    logger.debug("    i   norm_diff       delta")

    while iteration < max_iterations and \
            (x_prev is None or np.linalg.norm(x - x_prev) > min_variation):
        # Update delta
        delta = 0.5*(np.percentile(x, 75) - np.percentile(x, 25))
        if delta < min_delta:
            delta = min_delta

        # Update W
        Wi = W_inv(x, delta)

        # Advance x values by one iteration
        x_prev = x
        x = matrix_prod(
            Wi,
            phi_t,
            np.linalg.inv(matrix_prod(phi, Wi, phi_t) + lId),
            y
        )

        iteration += 1

        logger.debug(
            "{0:5d}  {1:10,.5f}  {2:10,.5f}".format(
                iteration, np.linalg.norm(x - x_prev), delta
            )
        )

    logger.info("Meridian CS done in " + str(iteration) + " iterations")
    
    return x

                        
