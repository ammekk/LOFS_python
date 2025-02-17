import numpy as np
from sklearn.preprocessing import StandardScaler


def standard_scale_data(data: np.ndarray, class_attribute_index: int) -> np.ndarray | None:
    if data.size == 0:
        return None

    scaler = StandardScaler()

    class_column = data[:, class_attribute_index].copy()

    data_to_scale = np.delete(data, class_attribute_index, axis=1)

    scaled_data = scaler.fit_transform(data_to_scale)

    return np.insert(scaled_data, class_attribute_index, class_column, axis=1)
