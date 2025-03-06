from typing import List, Tuple
import time as tm
import numpy as np

from LOFS_python.correlation_measure.fisher_z_test.my_cond_indep_fisher_z \
    import my_cond_indep_fisher_z


def saola_group_z_test(group_features: List[List[int]], data: np.ndarray, class_attribute: int,
                       alpha: float) -> Tuple[List[int], int, float]:
    start_time = tm.perf_counter()
    num_group = len(group_features)
    num_instances, num_features = data.shape
    current_feature = []
    dep = np.zeros(num_features)
    ci = 1
    g = [[] for _ in range(num_group)]
    f_index = list(range(num_features))

    for i in range(num_group):
        f_g_index: List[int] = group_features[i]
        current_feature = []

        for j in range(len(f_g_index)):
            # for very sparse data
            n1 = np.sum(data[:, f_g_index[j]])
            if n1 == 0:
                continue

            ci, dep[f_g_index[j]], _ = my_cond_indep_fisher_z(data, f_g_index[j], class_attribute,
                                                              [], num_instances, alpha)

            if ci == 1 or np.isnan(dep[f_g_index[j]]):
                continue

            current_feature.append(f_g_index[j])
            current_feature1 = [item for item in current_feature if item != f_g_index[j]]

            if current_feature1:
                p = len(current_feature1)

                for k in range(p):
                    ci, dep_ij, _ = my_cond_indep_fisher_z(data, f_g_index[j], current_feature1[k],
                                                           [], num_instances, alpha)

                    if ci == 1 or np.isnan(dep_ij):
                        continue

                    t_dep = dep_ij
                    t_feature = current_feature1[k]
                    if dep[t_feature] >= dep[f_g_index[j]] and t_dep > min(dep[f_g_index[j]], dep[t_feature]):
                        current_feature = [item for item in current_feature if item != f_g_index[j]]
                        break

                    if dep[f_g_index[j]] > dep[t_feature] and t_dep > min(dep[f_g_index[j]], dep[t_feature]):
                        current_feature = [item for item in current_feature if item != t_feature]

        g[i] = current_feature
        if g[i]:
            ci = 1

            for m in range(i - 1):
                g1 = g[m]

                for m1 in range(len(current_feature)):

                    for m2 in range(len(g1)):
                        ci, dep_ij1, _ = my_cond_indep_fisher_z(data, g1[m2], current_feature[m1],
                                                                [], num_instances, alpha)

                        if ci == 1 or np.isnan(dep_ij1):
                            continue

                        t_dep1 = dep_ij1
                        t_feature1 = current_feature[m1]

                        if dep[g1[m2]] > dep[t_feature1] and t_dep1 > min(dep[g1[m2]], dep[t_feature1]):
                            g[i] = [item for item in g[i] if item != t_feature1]
                            break

                        if dep[t_feature1] >= dep[g1[m2]] and t_dep1 > min(dep[g1[m2]], dep[t_feature1]):
                            g[m] = [item for item in g[m] if item != g1[m2]]

    select_feature = []
    select_group = 0

    for i in range(num_group):
        if g[i]:
            select_feature.extend(g[i])
            select_group += 1

    time = tm.perf_counter() - start_time
    return select_feature, select_group, time


