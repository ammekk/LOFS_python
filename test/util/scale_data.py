import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np
from sklearn.preprocessing import StandardScaler

import numpy as np
from sklearn.preprocessing import StandardScaler


def standard_scale_fast(data: np.ndarray, batch_size: int=1000) -> np.ndarray:
    if data is None or data.size == 0:
        raise ValueError("Error: Received None or empty array in batch_standard_scale!")

    scaler = StandardScaler()

    # Compute mean and std incrementally
    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i + batch_size, :]
        if batch.size == 0:
            raise ValueError(f"Empty batch encountered at index {i}!")
        scaler.partial_fit(batch)

    # Allocate space for scaled data
    scaled_data = np.zeros_like(data, dtype=np.float32)

    # Apply transformation in batches
    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i + batch_size, :]
        if batch.size == 0:
            raise ValueError(f"Empty batch encountered at index {i} during transformation!")

        transformed_batch = scaler.transform(batch)
        scaled_data[i:i + batch_size, :] = transformed_batch

    return scaled_data


def standard_scale_simple(data: np.ndarray, class_attribute_index: int = -1) -> np.ndarray | None:
    if data.size == 0:
        return None

    scaler = StandardScaler()

    if class_attribute_index == -1:
        return scaler.fit_transform(data)

    # Efficiently extract class column without copying
    class_column = data[:, class_attribute_index]

    # Avoid np.delete (expensive) by using slicing
    if class_attribute_index == 0:
        data_to_scale = data[:, 1:]  # Skip first column
    elif class_attribute_index == data.shape[1] - 1:
        data_to_scale = data[:, :-1]  # Skip last column
    else:
        data_to_scale = np.hstack((data[:, :class_attribute_index], data[:, class_attribute_index + 1:]))

    # Scale the features efficiently
    scaled_data = scaler.fit_transform(data_to_scale)

    # Insert class column back efficiently
    scaled_data = np.concatenate(
        (scaled_data[:, :class_attribute_index], class_column[:, None], scaled_data[:, class_attribute_index:]), axis=1)

    return scaled_data


def min_max_scale(data: np.ndarray, class_attribute_index: int = -1) -> np.ndarray:
    """
    Applies MinMaxScaler to all columns in X except the column at the class attribute index.
    """
    if data.size == 0:
        return None

    scaler = MinMaxScaler()

    if class_attribute_index == -1:
        return scaler.fit_transform(data)

    # Copy the class attribute column to preserve it
    class_column = data[:, class_attribute_index].copy()

    # Remove the class column before scaling
    data_to_scale = np.delete(data, class_attribute_index, axis=1)

    # Apply MinMaxScaler to the remaining data
    scaled_data = scaler.fit_transform(data_to_scale)

    # Reinsert the class column at its original position
    return np.insert(scaled_data, class_attribute_index, class_column, axis=1)
