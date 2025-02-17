from tokenize import String
from typing import List, Dict, Tuple, Any

import numpy as np


def print_selected_feature_info(select_features: List[int], time: float,
                                feature_map: Dict[int, str], select_groups: int = -1) -> None:
    select_feature_names = [feature_map[i] for i in select_features]
    if len(select_features) == 0:
        "Error: No features selected"
        return
    print(f"Time taken was {time} seconds")
    print(((f"The selected " + ("feature" if len(select_features) == 1 else "features") +
            f" came from {select_groups}" +
            (" group." if select_groups == 1 else " groups."))
           if select_groups != -1 else ""))
    prefix = "The selected feature is " if len(select_feature_names) == 1 else "The selected features are "
    names = (
        select_feature_names[0] if len(select_feature_names) == 1
        else f"{select_feature_names[0]} and {select_feature_names[1]}" if len(select_feature_names) == 2
        else f"{', '.join(select_feature_names[:-1])}, and {select_feature_names[-1]}"
    )
    print(prefix + names)


def print_data_info(attribute_names: List[Any], class_attribute_idx: int, data_shape: Tuple[int, int]) -> None:
    if len(attribute_names) >= 10:
        print("First 10 feature labels are " + " ".join(attribute_names[i] for i in range(10)))
    else:
        print(f"First {len(attribute_names)} " + " ".join(attribute_names[i] for i in range(len(attribute_names))))
    print("Class label is " + attribute_names[class_attribute_idx])
    print(f"Shape of data is {data_shape}")

def print_compare_feature_results(algo_name: String, test_name: String, selected_features_common: List[int],
                                  feature_map: dict[int, str]) -> None:
    attribute_common_names = [feature_map[idx] for idx in selected_features_common]
    print(f"There are {len(attribute_common_names)} between {algo_name} and {test_name}")
    if len(selected_features_common) > 0:
        prefix = "The feature in common is " if len(selected_features_common) == 1 else "The features in common are "
        names = (
            attribute_common_names[0] if len(attribute_common_names) == 1
            else f"{attribute_common_names[0]} and {attribute_common_names[1]}"
            if len(attribute_common_names) == 2
            else f"{', '.join(attribute_common_names[: -1])}, and {selected_features_common[-1]}"
        )
        print(prefix + names)

