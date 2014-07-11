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
noise_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
N = 300              
sample_divisions = 6

out_file = proj_utils.OUT_DIR / 'poly_6_noise_conv.csv'

def main():
    with open(out_file, 'w') as f:
        writer = csv.writer(f)
        modeler = functools.partial(
            cs_hdmr.cs_hdmr_modeler,
            ord_cap=3,
            deg_cap=6
        )

        errors = [
            model_validation.modeler_function_l2_errors(
                modeler, N*sample_divisions,
                model_validation.uniform_sampler(6, [-1.0, 1.0]),
                test_poly, sample_divisions=sample_divisions,
                noise=model_validation.make_gaussian_noise_fn(0, noise)
            )
            for noise in noise_list
        ]

        for noise, error_list in zip(noise_list, errors):
            writer.writerow([noise] + error_list)
    
if __name__ == '__main__':
    main()
