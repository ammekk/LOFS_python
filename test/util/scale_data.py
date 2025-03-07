import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np
from sklearn.preprocessing import StandardScaler

import numpy as np
from sklearn.preprocessing import StandardScaler


def min_max_scale_fast(data: np.ndarray, exclude_index: int = -1) -> np.ndarray:
    """
    Efficiently scales all columns in the dataset except the column at `exclude_index`.

    Parameters:
    - data (np.ndarray): The input dataset (2D NumPy array).
    - exclude_index (int): The column index to exclude from scaling.

    Returns:
    - np.ndarray: Scaled dataset with the excluded column left unchanged.
    """
    if data.size == 0:
        return None

    # If no column is excluded, scale the entire dataset directly
    if exclude_index == -1:
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)  # Scale everything efficiently

    # Create mask for selected columns (avoids np.delete for better efficiency)
    mask = np.ones(data.shape[1], dtype=bool)
    mask[exclude_index] = False

    # Use a view to avoid unnecessary memory copies
    data_to_scale = data[:, mask]

    # Apply MinMaxScaler once
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_to_scale)

    # Use in-place assignment to avoid extra copies
    result = data.astype(np.float64, copy=True)  # Ensure float type but avoid unnecessary conversion
    result[:, mask] = scaled_data  # Assign scaled values
    # No need to modify excluded column; it remains unchanged

    return result


def standard_scale_fast(data: np.ndarray, class_attribute_idx=-1, batch_size: int = 1000) -> np.ndarray:
    if data is None or data.size == 0:
        raise ValueError("Error: Received None or empty array in standard_scale_fast!")

    # If class_attribute_idx is -1, scale all columns
    if class_attribute_idx == -1:
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
            scaled_data[i:i + batch_size, :] = scaler.transform(batch)

        return scaled_data

    # Create a mask to exclude the specified column from scaling
    mask = np.ones(data.shape[1], dtype=bool)
    mask[class_attribute_idx] = False  # Exclude the class column

    # Initialize the scaler
    scaler = StandardScaler()

    # Compute mean and std on the selected columns incrementally
    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i + batch_size, mask]
        if batch.size == 0:
            raise ValueError(f"Empty batch encountered at index {i}!")
        scaler.partial_fit(batch)

    # Allocate space for scaled data

    # Apply transformation in batches
    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i + batch_size, mask]
        if batch.size == 0:
            raise ValueError(f"Empty batch encountered at index {i} during transformation!")

        data[i:i + batch_size, mask] = scaler.transform(batch)  # Scale only selected columns

    return data


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


def min_max_scale_simple(data: np.ndarray, class_attribute_index: int = -1) -> np.ndarray:
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
