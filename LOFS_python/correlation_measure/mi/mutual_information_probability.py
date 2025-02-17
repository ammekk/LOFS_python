import numpy as np
from LOFS_python.correlation_measure.mi.calculate_probability import calculate_joint_probability
from LOFS_python.correlation_measure.mi.entropy import calculate_conditional_entropy
from LOFS_python.correlation_measure.mi.array_operations import merge_arrays
from LOFS_python.correlation_measure.mi.probability_state_classes import JointProbabilityState


def calculate_mutual_information(data_vector: np.ndarray, target_vector: np.ndarray, vector_length: int) -> float:
    """
    Computes the Mutual Information (MI) between two discrete variables.

    Mutual Information quantifies the amount of information obtained about
    one variable through the other. It is calculated using the probability
    distributions of individual and joint states of the input variables.

    Formula:
        I(X;Y) = Σ Σ p(x,y) * log2( p(x,y) / (p(x) * p(y)) )
    """
    state = calculate_joint_probability(data_vector, target_vector, vector_length)
    mutual_information = np.sum(
        state.joint_probability_vector[i] * np.log(
            state.joint_probability_vector[i] /
            (state.first_probability_vector[i % state.num_first_states] *
             state.second_probability_vector[i // state.num_first_states])
        )
        for i in range(state.num_joint_states)
        if state.joint_probability_vector[i] > 0
    ) / np.log(2.0)

    return mutual_information


def calculate_conditional_mutual_information(data_vector: np.ndarray, target_vector: np.ndarray,
                                             conditional_vector: np.ndarray, vector_length: int) -> float:
    """
    Computes the Conditional Mutual Information (CMI) between three discrete variables.

    Conditional Mutual Information quantifies the dependency between two variables
    while controlling for a third variable. It is computed as:
        I(X;Y|Z) = H(X|Z) - H(X|YZ)
    """
    merged_vector = np.zeros(vector_length, dtype=np.float64)

    merge_arrays(target_vector, conditional_vector, merged_vector, vector_length)

    first_condition = calculate_conditional_entropy(data_vector, conditional_vector, vector_length)
    second_condition = calculate_conditional_entropy(data_vector, merged_vector, vector_length)

    mutual_information = first_condition - second_condition
    return mutual_information


