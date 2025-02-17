import numpy as np
from LOFS_python.correlation_measure.mi.calculate_probability import calculate_probability, calculate_joint_probability
from LOFS_python.correlation_measure.mi.probability_state_classes import ProbabilityState, JointProbabilityState


def calculate_entropy(data_vector: np.ndarray, vector_length: int) -> float:
    """
    Computes the Shannon entropy H(X) of a discrete random variable.
    """
    entropy: float = 0.0
    state: ProbabilityState  = calculate_probability(data_vector, vector_length)

    for i in range(state.num_states):
        temp_value: float = state.probability_vector[i]

        if temp_value > 0:
            entropy -= temp_value * np.log(temp_value)

    entropy /= np.log(2.0)

    return entropy


def calculate_joint_entropy(first_vector: np.ndarray, second_vector: np.ndarray, vector_length: int) -> float:
    """
    Computes the joint entropy H(X,Y) of two discrete random variables.
    """
    joint_entropy = 0.0
    state: JointProbabilityState = calculate_joint_probability(first_vector, second_vector, vector_length)

    for i in range(state.num_joint_states):
        temp_value = state.joint_probability_vector[i]
        if temp_value > 0:
            joint_entropy -= temp_value * np.log(temp_value)

    joint_entropy /= np.log(2.0)
    return joint_entropy


def calculate_conditional_entropy(data_vector: np.ndarray, condition_vector: np.ndarray, vector_length: int) -> float:
    """
    Computes the conditional entropy H(X|Y) of two discrete random variables.
    """
    state = calculate_joint_probability(data_vector, condition_vector, vector_length)
    cond_entropy = -sum(
        p * np.log(p / state.second_probability_vector[i // state.num_first_states])
        for i, p in enumerate(state.joint_probability_vector) if p > 0
    ) / np.log(2.0)

    return cond_entropy


