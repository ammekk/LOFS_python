from typing import Callable

import numpy as np
from LOFS_python.correlation_measure.mi.mi_toolbox_mex import mex_function


def mi(x: np.ndarray, y: np.ndarray) -> float:
    """
    x & y can be matrices which are converted into a joint variable
    before computation

    expects variables to be column-wise

    returns the mutual information between X and Y, I(X;Y)
    """
    merge: Callable[[np.ndarray], np.ndarray] = \
        lambda a: mex_function(3, a) if a.shape[1] > 1 else a
    return mex_function(7, merge(x), merge(y))
