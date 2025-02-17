import numpy as np
from LOFS_python.correlation_measure.mi.probability_state_classes import ProbabilityState, JointProbabilityState
from LOFS_python.correlation_measure.mi.array_operations import normalise_array


def calculate_probability(data_vector: np.ndarray, vector_length: int) -> ProbabilityState:
    """
    Computes the probability distribution of discrete states in a given data vector.

    This function normalizes the input data, counts occurrences of each unique state,
    and calculates the probability of each state based on the total vector length.
    """
    normalised_vector = np.zeros(vector_length, dtype=np.int32)
    num_states = normalise_array(data_vector, normalised_vector, vector_length)

    state_counts = np.bincount(normalised_vector, minlength=num_states)
    state_probs = state_counts / vector_length

    return ProbabilityState(state_probs, num_states)


def calculate_joint_probability(first_vector: np.ndarray, second_vector: np.ndarray,
                                vector_length: int) -> JointProbabilityState:
    """
    Computes the joint probability distribution of two discrete variables.

    This function normalizes the input vectors, counts occurrences of each unique state,
    and calculates the probability distribution of the joint states.
    """
    first_normalized_vector = np.zeros(vector_length,
                                       dtype=np.int64)
    second_normalized_vector = np.zeros(vector_length, dtype=np.int64)

    first_num_states = normalise_array(first_vector, first_normalized_vector, vector_length)
    second_num_states = normalise_array(second_vector, second_normalized_vector, vector_length)
    joint_num_states = first_num_states * second_num_states

    first_state_counts = np.bincount(first_normalized_vector, minlength=first_num_states)
    second_state_counts = np.bincount(second_normalized_vector, minlength=second_num_states)
    joint_state_counts = np.bincount(second_normalized_vector * first_num_states
                                     + first_normalized_vector, minlength=joint_num_states)

    first_state_probs = first_state_counts / vector_length
    second_state_probs = second_state_counts / vector_length
    joint_state_probs = joint_state_counts / vector_length

    return JointProbabilityState(joint_state_probs, joint_num_states, first_state_probs, first_num_states,
                                 second_state_probs, second_num_states)
