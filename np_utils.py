import numpy as np

def array_map(f, inputs):
    return np.apply_along_axis(
        f, axis=1, arr=inputs
    )
