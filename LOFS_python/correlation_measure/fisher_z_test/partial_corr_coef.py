from typing import List, Tuple

import numpy as np


def partial_corr_coef(s: np.ndarray, i: int, j: int, y: np.ndarray) -> Tuple[float, float]:
    s_y_y = s[np.ix_(y, y)]  # Extract only once
    s_x_y = s[np.ix_([i, j], y)]  # Extract only once
    s_x_x = s[np.ix_([i, j], [i, j])]

    # Solve the system instead of computing matrix inverse (faster & numerically stable)
    s2 = s_x_x - s_x_y @ np.linalg.solve(s_y_y, s_x_y.T)

    c = s2[0, 1]
    r = c / np.sqrt(s2[0, 0] * s2[1, 1])
    return r, c
