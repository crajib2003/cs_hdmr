import model_validation
import cs_hdmr
import functools
import logging
import cProfile
import proj_utils
import csv

logging.basicConfig()

def test_poly(x1, x2, x3, x4, x5, x6):
    return (x1)*(x2)*(x5) + (x2)*(x3) + (x4)*(x3**2) + x6
n_list = [400]
sample_divisions = 2

out_file = proj_utils.OUT_DIR / 'poly_6_conv.csv'

def main():
    with open(out_file, 'w') as f:
        writer = csv.writer(f)
        modeler = functools.partial(
            cs_hdmr.cs_hdmr_modeler,
            ord_cap=3,
            deg_cap=6,
            independent=True
        )

        errors = [
            model_validation.modeler_function_l2_errors(
                modeler, n*sample_divisions,
                model_validation.uniform_sampler(6, [-1.0, 1.0]),
                test_poly, sample_divisions=sample_divisions
            )
            for n in n_list
        ]

        for n, error_list in zip(n_list, errors):
            writer.writerow([n] + error_list)
    
if __name__ == '__main__':
    main()
