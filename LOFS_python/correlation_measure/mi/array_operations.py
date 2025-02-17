import numpy as np


def increment_vector(vector: np.ndarray) -> None:
    vector += 1


def print_vector(vector: np.ndarray, vector_length: int) -> None:
    for i in range(vector_length):
        print(f"Val at i={i}, is {vector[i]}")


def number_of_unique_values(feature_vector: np.ndarray) -> int:
    """
    Count unique values and replace each value with its order of appearance (starting from 1).
    Modifies feature_vector in-place and returns the number of unique values found.
    """
    unique_values, indices = np.unique(feature_vector, return_inverse=True)
    feature_vector[:] = indices + 1
    return len(unique_values)


def normalise_array(input_vector: np.ndarray, output_vector: np.ndarray, vector_length: int) -> int:
    """
    Normalise_array takes an input vector and writes an output vector
    which is a normalised version of the input, and returns the number of states
    A normalised array has min value = 0, max value = number of states
    and all values are integers
    length(inputVector) == length(outputVector) == vectorLength
    """
    input_vector = input_vector.ravel()

    if vector_length == 0:
        return 1

    unique_values, normalized_indices = np.unique(input_vector, return_inverse=True)

    output_vector[:] = normalized_indices

    return len(unique_values)


def merge_arrays(first_vector: np.ndarray, second_vector: np.ndarray, output_vector: np.ndarray,
                 vector_length: int) -> int:
    """
    Merge_arrays takes in two arrays and writes the joint state of those arrays
    to the output vector, returning the number of joint states.
    The length of the vectors must be the same and equal to vectorLength
    """
    first_normalized_vector = np.zeros(vector_length, dtype=np.int32)
    second_normalized_vector = np.zeros(vector_length, dtype=np.int32)

    first_num_states = normalise_array(first_vector, first_normalized_vector, vector_length)
    second_num_states = normalise_array(second_vector, second_normalized_vector, vector_length)

    state_map = np.zeros(first_num_states * second_num_states, dtype=np.int32)
    state_count = 1

    for i in range(vector_length):
        cur_index = first_normalized_vector[i] + (second_normalized_vector[i] * first_num_states)
        if state_map[cur_index] == 0:
            state_map[cur_index] = state_count
            state_count += 1
        output_vector[i] = state_map[cur_index]

    return state_count


# Omitting vector_length from args because its mot needed
def merge_arrays_arities(first_vector: np.ndarray, num_first_states: int, second_vector: np.ndarray,
                         num_second_states: int, output_vector: np.ndarray) -> int:
    """
    Combines two vectors into joint states, checking against expected number of states.
    Returns total number of states if successful, -1 if state limits exceeded.
    """
    total_states = -1
    first_normalized_vector = np.zeros_like(first_vector, dtype=np.int32)
    second_normalized_vector = np.zeros_like(first_vector, dtype=np.int32)

    first_state_check = normalise_array(first_vector, first_normalized_vector)
    second_state_check = normalise_array(second_vector, second_normalized_vector)

    if first_state_check <= num_first_states and second_state_check <= num_second_states:
        output_vector[:] = first_normalized_vector + (second_normalized_vector * num_first_states) + 1
        total_states = num_first_states * num_second_states

    return total_states


def merge_multiple_arrays(input_matrix: np.ndarray, output_vector: np.ndarray, matrix_width: int,
                          vector_length: int) -> int:
    """
    This function takes an input matrix representing multiple feature vectors,
    iteratively merges them column by column into a single output vector, and returns
    the number of unique joint states.
    """
    # Flattening it allows it to work like the c code with any vector length
    input_matrix = input_matrix.ravel()
    current_num_states = 0

    if matrix_width > 1:
        current_num_states = merge_arrays(input_matrix[:vector_length],
                                               input_matrix[vector_length:2 * vector_length], output_vector,
                                               vector_length)

        for i in range(2, matrix_width):
            current_index = i * vector_length
            current_num_states = merge_arrays(output_vector,
                                              input_matrix[current_index: current_index + vector_length],
                                              output_vector, vector_length)

    else:
        normalized_vector = np.zeros(vector_length, dtype=np.int32)
        current_num_states = normalise_array(input_matrix, normalized_vector, vector_length)

        output_vector[:] = input_matrix[:vector_length]

    return current_num_states


def merge_multiple_arrays_arities(input_matrix: np.ndarray, output_vector: np.ndarray, matrix_width: int,
                                  arities: np.ndarray, vector_length: int) -> int:
    """
    Merges multiple input vectors with known arities into a single joint state vector.
    This function processes a flattened 1D input matrix, merges its feature vectors column by column,
    and ensures that the joint states are assigned correctly based on the given arities.
    """
    # Flattening it allows it to work like the c code with any vector length
    input_matrix = input_matrix.ravel()
    current_num_states = 0

    if matrix_width > 1:
        current_num_states = merge_arrays_arities(input_matrix[:vector_length], arities[0],
                                                  input_matrix[vector_length: vector_length * 2], arities[1],
                                                  output_vector)

        for i in range(2, matrix_width):
            current_index = i * vector_length
            current_num_states = merge_arrays_arities(output_vector, current_num_states,
                                                      input_matrix[current_index:current_index + vector_length],
                                                      arities[i], output_vector)
            if current_num_states == -1:
                break

    else:
        normalized_vector = np.zeros(vector_length, dtype=np.int32)
        current_num_states = normalise_array(input_matrix, normalized_vector, vector_length)

        output_vector[:] = input_matrix[:vector_length]

    return current_num_states
