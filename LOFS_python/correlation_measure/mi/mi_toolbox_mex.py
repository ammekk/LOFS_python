import numpy as np
from typing import Union, List, Tuple
from LOFS_python.correlation_measure.mi.calculate_probability import calculate_probability, calculate_joint_probability
from LOFS_python.correlation_measure.mi.array_operations import merge_multiple_arrays, merge_multiple_arrays_arities, normalise_array
from LOFS_python.correlation_measure.mi.entropy import calculate_entropy, calculate_joint_entropy, calculate_conditional_entropy
from LOFS_python.correlation_measure.mi.probability_state_classes import *
from LOFS_python.correlation_measure.mi.mutual_information_probability import calculate_mutual_information, calculate_conditional_mutual_information


def mex_function(*prhs: Union[int, np.ndarray]) -> Union[Tuple[np.ndarray, int],
                        Tuple[np.ndarray, float, np.ndarray, float, np.ndarray, float], float, np.ndarray,
                        Tuple[np.ndarray, np.ndarray], None]:
    """
    Differs from c version in inputs and outputs
    Parameters:
        *prhs: an int followed by at least one np.ndarray. The first int item represents the flag needed for the match
    Returns:
        output: depending on the case output can be a tuple holding a combination of ints, np.ndarray, ar ndarray or a float. It
        differs from the c code in that it returns None instead of a 2d array
        containing just -1 if it errors.
    Note:
        ravel() used to flatten np.ndarrays to make them more like c matrixes
    """
    nrhs = len(prhs)

    if nrhs not in {1, 2, 3, 4}:
        print("Incorrect number of arguments, format is MIToolbox(\"FLAG\",varargin)\n")
        return None

    flag = prhs[0]

    match flag:
        # case 1: calculate probability
        case 1:
            number_of_samples = prhs[1].shape[0]
            data_vector: np.ndarray = prhs[1].astype(np.float64).ravel()
            state: ProbabilityState = calculate_probability(data_vector, number_of_samples)

            return state.probability_vector.reshape(-1, 1), state.num_states

        # case 2 - calculate joint probability
        case 2:
            number_of_samples = prhs[1].shape[0]
            # Ensures np.ndarray is treated like a c matrix
            first_vector = prhs[1].astype(np.float64).ravel()
            second_vector = prhs[2].astype(np.float64).ravel()

            joint_state: JointProbabilityState = \
                calculate_joint_probability(first_vector, second_vector, number_of_samples)

            joint_output = joint_state.joint_probability_vector.reshape(-1, 1)
            first_output = joint_state.first_probability_vector.reshape(-1, 1)
            second_output = joint_state.second_probability_vector.reshape(-1, 1)

            return (joint_output, float(joint_state.num_joint_states),
                    first_output, float(joint_state.num_first_states),
                    second_output, float(joint_state.num_second_states))

        # case 3 - merge arrays
        case 3:
            number_of_samples, number_of_features = prhs[1].shape
            num_arities = prhs[2].shape[1] if nrhs > 2 else 0

            output = None

            if num_arities == 0:
                if number_of_features > 0 and number_of_samples > 0:
                    merged_vector = np.zeros(number_of_samples, dtype=np.float64)
                    merge_multiple_arrays(prhs[1], merged_vector, number_of_features, number_of_samples)
                    output = merged_vector.reshape(-1, 1)

            elif num_arities == number_of_features:
                if number_of_features > 0 and number_of_samples > 0:
                    merged_vector = np.zeros(number_of_samples, dtype=np.float64)
                    int_arities = np.floor(prhs[2].ravel()).astype(np.int32)

                    if merge_multiple_arrays_arities(prhs[1], merged_vector, int_arities, number_of_samples) != -1:
                        output = merged_vector.reshape(-1, 1)

            else:
                print("Number of arities does not match number of features, arities should be a row vector")

            return output

        # case 4 - H(X)
        case 4:
            number_of_samples, number_of_features = prhs[1].shape
            return calculate_entropy(prhs[1].ravel(), number_of_samples)\
                if number_of_features == 1 else None

        # case 5 - H(XY)
        case 5:
            number_of_samples, number_of_features = prhs[1].shape
            check_samples, check_features = prhs[2].shape

            first_vector = prhs[1].ravel()
            second_vector = prhs[2].ravel()

            if number_of_features == 1 and check_features == 1:
                output = (
                    0.0 if number_of_samples == check_samples == 0 else
                    calculate_entropy(second_vector, number_of_samples) if number_of_samples == 0 else
                    calculate_entropy(first_vector, number_of_samples) if check_samples == 0 else
                    calculate_joint_entropy(first_vector, second_vector, number_of_samples)
                    if number_of_samples == check_samples else None
                )

                if output is None:
                    print("Vector lengths do not match, they must be the same length")

            else:
                output = None
                print("No columns in input")

            return output

        # case 6 - H(X|Y)*
        case 6:
            number_of_samples, number_of_features = prhs[1].shape
            check_samples, check_features = prhs[2].shape

            data_vector = prhs[1].ravel()
            cond_vector = prhs[2].ravel()

            if number_of_features == 1 and check_features == 1:
                output = (
                    0.0 if number_of_samples == 0 else
                    calculate_entropy(data_vector, number_of_samples) if check_samples == 0 else
                    calculate_conditional_entropy(data_vector, cond_vector, number_of_samples)
                    if number_of_samples == check_samples else None
                )
                if output is None:
                    print("Vector lengths do not match, they must be the same length")
            else:
                output = None
                print("No columns in input")

            return output
        # case 7 - I(X;Y)
        case 7:
            number_of_samples, number_of_features = prhs[1].shape
            check_samples, check_features = prhs[2].shape

            first_vector = prhs[1].ravel()
            second_vector = prhs[2].ravel()

            if number_of_features == 1 and check_features == 1:
                output = (
                    0.0 if number_of_samples == 0 or check_samples == 0 else
                    calculate_mutual_information(first_vector, second_vector, number_of_samples)
                    if number_of_samples == check_samples else None
                )
                if output is None:
                        print("Vector lengths do not match, they must be the same length")
            else:
                output = None
                print("No columns in input")

            return output
        # case 8 - I(X;Y|Z
        case 8:
            (number_of_samples, number_of_features) = prhs[1].shape
            (check_samples, check_features) = prhs[2].shape
            (third_check_samples, third_check_features) = prhs[3].shape

            first_vector = prhs[1].ravel()
            target_vector = prhs[2].ravel()
            cond_vector = prhs[3].ravel()

            if number_of_features == check_features == 1:
                output = (
                    0.0 if number_of_samples == 0 or check_samples == 0 else
                    calculate_mutual_information(first_vector, target_vector, number_of_samples)
                    if third_check_samples == 0 or third_check_features != 1 else
                    calculate_conditional_mutual_information(first_vector, target_vector, cond_vector,
                                                             number_of_samples)
                    if number_of_samples == check_samples == third_check_samples else None
                )
            if output is None:
                    output = None
                    print("Vector lengths do not match, they must be the same lengths")
            else:
                output = None
                print("No columns in input")

            return output
        # case 9 - normaliseArray
        case 9:
            number_of_samples = prhs[1].shape[0]

            data_vector = prhs[1].ravel()
            output_int_vector = np.zeros((number_of_samples, 1), dtype=np.int32)

            num_states = normalise_array(data_vector, output_int_vector, number_of_samples)

            return output_int_vector.astype(np.float64), num_states

        case _:
            print("Unrecognized flag")
            return None

