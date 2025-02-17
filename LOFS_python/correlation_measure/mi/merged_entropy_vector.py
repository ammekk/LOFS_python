from LOFS_python.correlation_measure.mi.mi_toolbox_mex import *
import numpy as np


def h(x: np.ndarray) -> float:
    """
    can be a matrix which is converted into a joint variable before calculation
    expects variables to be column-wise

    returns the entropy of X, H(X)
    """
    return mex_function(4, mex_function(3, x) if x.shape[1] > 1 else x)
