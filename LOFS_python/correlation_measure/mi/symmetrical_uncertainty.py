import numpy as np

from LOFS_python.correlation_measure.mi.merged_entropy_vector import h
from LOFS_python.correlation_measure.mi.mutual_information_matrix_analyzer \
    import mi


def su(first_vector: np.ndarray, second_vector: np.ndarray) -> float:
    """
    Calculates the  (SU) or symmetrical uncertainty between two variables.
    SU is a normalized version of mutual information that ranges from 0 to 1,
    where: SU(X,Y) = 2 * I(X;Y) / (H(X) + H(Y))
    """
    return (2 *
            mi(first_vector, second_vector)) \
        / (h(first_vector)
           + h(second_vector))
