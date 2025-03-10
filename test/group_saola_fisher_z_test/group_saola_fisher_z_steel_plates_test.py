import numpy as np
import openml
from LOFS_python.learning_module.group_saola.group_f import group_features
from LOFS_python.learning_module.group_saola.saola_group_z_test import saola_group_z_test
from test.test_selected_features.model_comparrison import random_forest_classifier_acc_val, pca
from test.util.get_feature_collections import extract_idx_to_feature_map, extract_feature_index_list, extract_same_features
from test.util.scale_data import min_max_scale_simple
from test.util.visualize import print_selected_feature_info, print_data_info, print_compare_feature_results
from test.test_selected_features.alt_function_selection_methods \
    import select_top_k_features_anova_f_statistic, select_top_k_features_mutual_information, \
    random_forest_classifier_select_features

# data set from https://openml.org/search?type=data&sort=runs&status=active&id=1504
# Both versions work well though fisher_z works slightly better which makes sense because
# data is continuous
def main():
    dataset = openml.datasets.get_dataset(1504, download_features_meta_data=True)
    x, _, _, attribute_names = dataset.get_data()
    data = x.to_numpy()
    class_attribute_idx = len(attribute_names) - 1
    y = data[:, class_attribute_idx]
    data = min_max_scale_simple(data, class_attribute_idx)

    print_data_info(data, attribute_names, class_attribute_idx, data.shape)

    feature_map = extract_idx_to_feature_map(attribute_names, class_attribute_idx)
    feature_index = extract_feature_index_list(attribute_names, class_attribute_idx,
                                               lambda a: int(a[1:]) - 1)

    group_feature = group_features(feature_index, 3)

    select_features, select_groups, time = saola_group_z_test(group_feature, data.astype(np.float64),
                                                           class_attribute_idx, .001)
    print_selected_feature_info(select_features, time, feature_map, "Group-saola fisher-z test", select_groups)

    anova_features = select_top_k_features_anova_f_statistic(data[:, :-1], data[:, -1], feature_map,
                                                             len(select_features))
    print_compare_feature_results("group-saola", "anova-f-statistic",
                                  extract_same_features(select_features, anova_features), feature_map)

    mutual_information_features = select_top_k_features_mutual_information(data[:, :-1], data[:, -1], feature_map,
                                                                           len(select_features))
    print_compare_feature_results("group-saola", "mutusl-information",
                                  extract_same_features(select_features, mutual_information_features), feature_map)

    random_forest_features = random_forest_classifier_select_features(data[:, :-1], data[:, -1], feature_map,
                                                                      k=len(select_features))
    print_compare_feature_results("group-saola", "random_forest",
                                  extract_same_features(select_features, random_forest_features), feature_map)

    random_forest_classifier_acc_val(data[:, :-1], data[:, -1], select_features)
    pca(data[:, :-1], data[:, -1], len(select_features))



if __name__ == "__main__":
    main()