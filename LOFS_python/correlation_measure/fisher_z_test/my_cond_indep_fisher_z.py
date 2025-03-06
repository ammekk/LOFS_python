import math
from typing import List, Tuple
from LOFS_python.correlation_measure.fisher_z_test.partial_corr_coef import partial_corr_coef

import numpy as np
from scipy.stats import norm


def my_cond_indep_fisher_z(data: np.ndarray, x: int, y: int, s: List[int], n: int,
                           alpha: float = 0.05) -> Tuple[int, float, float]:

    cols = [x, y] + s
    data_subset = np.take(data, cols, axis=1)

    c = np.cov(data_subset, rowvar=False, dtype=np.float32)

    size_c = c.shape[1]
    s1 = np.arange(2, size_c)

    r, _ = partial_corr_coef(c, 0, 1, s1)
    r = np.clip(r, -0.999999, 0.999999)

    z = math.atanh(r)
    w = math.sqrt(n - len(s1) - 3) * z
    cutoff = norm.ppf(1 - 0.5 * alpha)

    ci = int(abs(w) < cutoff)

    p = norm.cdf(w)
    r = abs(r)

    return ci, r, p




