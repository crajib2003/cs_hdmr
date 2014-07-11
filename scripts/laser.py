import proj_utils
import numpy as np
import cs_hdmr
import functools
import numpy.random
import model_validation
import logging

logging.basicConfig()

DATA_FILE = proj_utils.DATA_DIR / 'laser.txt'
a = np.loadtxt(DATA_FILE, skiprows=0)
np.random.shuffle(a)
a = a[:1000]
a[:,-1] *= 10000


modeler = functools.partial(
    cs_hdmr.cs_hdmr_modeler, ord_cap=4, deg_cap=5,
    independent=False
)

print model_validation.modeler_l2_errors(
    modeler, a, sample_divisions=2
)
