import numpy as np
import time as tm
from typing import List, Tuple
from LOFS_python.correlation_measure.fisher_z_test.my_cond_indep_fisher_z import my_cond_indep_fisher_z

cache_fisher_z = {}
def my_cond_indep_fisher_z_cached(data: np.ndarray, x: int, y: int, s: List[int], n: int, alpha: float = 0.05):
    key = (x, y, tuple(s))
    if key not in cache_fisher_z:
        cache_fisher_z[key] = my_cond_indep_fisher_z(data, x, y, s, n, alpha)
    return cache_fisher_z[key]


def saola_group_z_test(group_features: List[List[int]], data: np.ndarray, class_attribute: int,
                       alpha: float) -> Tuple[List[int], int, float]:
    start_time = tm.perf_counter()
    num_group = len(group_features)
    num_instances, num_features = data.shape

    dep = np.zeros(num_features, dtype=np.float32)  # Use float32 to save memory
    g = [list() for _ in range(num_group)]  # Keep lists for controlled order

    # First Pass: Feature Selection Within Groups
    for i, f_g_index in enumerate(group_features):
        current_feature = []

        for f_idx in f_g_index:
            if not np.any(data[:, f_idx]):  # 🚀 Fast check for nonzero values
                continue

            ci, dep[f_idx], _ = my_cond_indep_fisher_z_cached(data, f_idx, class_attribute, [], num_instances, alpha)
            if ci == 1 or np.isnan(dep[f_idx]):
                continue

            current_feature.append(f_idx)

            # Check dependencies within selected features
            for prev_feature in current_feature[:-1]:  # Ensure controlled order
                ci, dep_ij, _ = my_cond_indep_fisher_z_cached(data, f_idx, prev_feature, [], num_instances, alpha)
                if ci == 1 or np.isnan(dep_ij):
                    continue

                # Prune weaker features
                if dep[prev_feature] >= dep[f_idx] and dep_ij > min(dep[f_idx], dep[prev_feature]):
                    current_feature.remove(f_idx)  # Back to list-based removal
                    break
                if dep[f_idx] > dep[prev_feature] and dep_ij > min(dep[f_idx], dep[prev_feature]):
                    current_feature.remove(prev_feature)

        g[i] = current_feature  # Maintain list structure

    # Second Pass: Cross-Group Feature Selection
    for i in range(num_group):
        if not g[i]:
            continue

        for j in range(i):
            g1 = g[j]

            for feature in list(g[i]):  # Iterate over a copy to allow modification
                for other_feature in list(g1):  # Keep order stable
                    ci, dep_ij1, _ = my_cond_indep_fisher_z_cached(data, other_feature, feature, [], num_instances, alpha)
                    if ci == 1 or np.isnan(dep_ij1):
                        continue

                    # Prune weaker features
                    if dep[other_feature] > dep[feature] and dep_ij1 > min(dep[other_feature], dep[feature]):
                        g[i].remove(feature)  # Use list remove() to ensure order stability
                        break
                    if dep[feature] >= dep[other_feature] and dep_ij1 > min(dep[other_feature], dep[other_feature]):
                        g[j].remove(other_feature)

    # Final Selection
    select_feature = [
        f for i, group in enumerate(g) if group  # Ensure group is not empty
        for f in group if f != class_attribute  # Ensure class_attribute is not selected
    ]
    select_group = sum(1 for group in g if group)

    time_taken = tm.perf_counter() - start_time
    return select_feature, select_group, time_taken
