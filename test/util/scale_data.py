import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def standard_scale_data(data: np.ndarray, class_attribute_index: int = -1) -> np.ndarray | None:
    if data.size == 0:
        return None

    scaler = StandardScaler()

    if class_attribute_index == -1:
        return scaler.fit_transform(data)

    class_column = data[:, class_attribute_index].copy()

    data_to_scale = np.delete(data, class_attribute_index, axis=1)

    scaled_data = scaler.fit_transform(data_to_scale)

    return np.insert(scaled_data, class_attribute_index, class_column, axis=1)


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

