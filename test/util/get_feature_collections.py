from tokenize import String
from typing import List, Callable, Any, Union

import numpy as np


def extract_idx_to_feature_map(attribute_names: list[Any], class_attribute_idx: Any) -> dict[int, any]:
    return {idx: item for idx, item in enumerate(attribute_names) if idx != class_attribute_idx}


def extract_feature_index_list(attribute_names: List[Any], class_attribute_idx: int,
                               attribute_to_idx: Union[Callable[[Any], int], None]= None) -> List[int]:
    if not attribute_to_idx:
        return [i for i in range(len(attribute_names)) if i != class_attribute_idx]
    return [attribute_to_idx(item) for item in attribute_names if item != attribute_names[class_attribute_idx]]


def extract_same_features(original_features_selected: List[int], test_features_selected: List[int]) -> List[int]:
    return np.intersect1d(original_features_selected, test_features_selected).tolist()


def extract_idx_to_feature(feature_map: dict[int, str], feature_idx_list: List[int]) -> List[any]:
    return [feature_map[idx] for idx in feature_idx_list]
