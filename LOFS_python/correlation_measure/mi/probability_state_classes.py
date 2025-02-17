# Definition of classes mentioned in multiple places in this package
import numpy as np


class ProbabilityState:
    def __init__(self, probability_vector: np.ndarray, num_states: int):
        self.probability_vector = probability_vector
        self.num_states = num_states


class JointProbabilityState:
    def __init__(self, joint_probability_vector: np.ndarray, num_joint_states: int,
                 first_probability_vector: np.ndarray, num_first_states: int, second_probability_vector: np.ndarray,
                 num_second_states: int):
        self.joint_probability_vector = joint_probability_vector
        self.num_joint_states = num_joint_states
        self.first_probability_vector = first_probability_vector
        self.num_first_states = num_first_states
        self.second_probability_vector = second_probability_vector
        self.num_second_states = num_second_states

