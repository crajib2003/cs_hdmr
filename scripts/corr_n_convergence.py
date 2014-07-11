import model_validation
import cs_hdmr
import functools
import logging
import cProfile
import proj_utils
import csv
import tests.test_functions as test_functions
import numpy as np
import model_validation

logging.basicConfig()

mu = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
cov = np.array(
    [
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ]
)
sample_fn = model_validation.multi_gaussian_sampler(mu, cov)
test_poly = test_functions.gen_corr_poly_1(mu)

n_list = [140]
sample_divisions = 12

out_file = proj_utils.OUT_DIR / 'corr_conv.csv'

def main():
    with open(out_file, 'w') as f:
        writer = csv.writer(f)
        modeler = functools.partial(
            cs_hdmr.cs_hdmr_modeler,
            ord_cap=2,
            deg_cap=5
        )

        errors = [
            model_validation.modeler_function_l2_errors(
                modeler, n*sample_divisions, sample_fn,
                test_poly, sample_divisions=sample_divisions
            )
            for n in n_list
        ]

        for n, error_list in zip(n_list, errors):
            writer.writerow([n] + error_list)
    
if __name__ == '__main__':
    main()
