from tokenize import String
from typing import List, Callable, Any

import numpy as np


def extract_feature_map(attribute_names: list[Any], class_attribute_idx: Any) -> dict[int, str]:
    return {idx: item for idx, item in enumerate(attribute_names) if item != attribute_names[class_attribute_idx]}


def extract_feature_index_list(attribute_names: List[Any], class_attribute_idx: int,
                               attribute_to_idx: Callable[[Any], int]) -> List[int]:
    return [attribute_to_idx(item) for item in attribute_names if item != attribute_names[class_attribute_idx]]


def extract_same_features(original_features_selected: List[int], test_features_selected: List[int]) -> List[int]:
    print(original_features_selected)
    print(test_features_selected)
    return np.intersect1d(original_features_selected, test_features_selected).tolist()
