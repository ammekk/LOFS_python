from math import floor
from typing import List

import numpy as np


def group_features(features_index: List[int], number_groups: int) -> List[List[int]]:
    k = floor(len(features_index)/number_groups)
    group_feature = [[] for _ in range(number_groups)]

    j = 0
    for i in range(number_groups):
        if i != number_groups:
            group_feature[i] = features_index[j:j + k]

        if i == number_groups - 1:
            group_feature[i] = features_index[j:]

        j += k

    return group_feature
