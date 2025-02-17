import numpy as np
import openml

# data set from https://openml.org/search?type=data&sort=runs&status=active&id=1504


from LOFS_python.learning_module.group_saola.group_f import group_features
from LOFS_python.learning_module.group_saola.saola_group_mi_algo import saola_group_mi
from test.util.get_feature_collections import extract_feature_map, extract_feature_index_list, extract_same_features
from test.util.scale_data import standard_scale_data
from test.util.visualize import print_selected_feature_info, print_data_info, print_compare_feature_results
from test.alternative_feature_selection_methods.alternative_feature_selection_methods \
    import select_top_k_features_anova_f_statistic


def main():
    dataset = openml.datasets.get_dataset(1504, download_features_meta_data=True)
    x, _, _, attribute_names = dataset.get_data()
    data = x.to_numpy()
    class_attribute_idx = len(attribute_names) - 1

    print_data_info(attribute_names, class_attribute_idx, data.shape)



    feature_map = extract_feature_map(attribute_names, class_attribute_idx)
    feature_index = extract_feature_index_list(attribute_names, class_attribute_idx,
                                               lambda a: int(a[1:]) - 1)

    # getting features into groups
    group_feature = group_features(feature_index, 2)
    scaled_data = standard_scale_data(data, class_attribute_idx)

    select_features, select_groups, time = saola_group_mi(group_feature, scaled_data.astype(np.float64), class_attribute_idx, 0)
    print_selected_feature_info(select_features, time, feature_map, select_groups)

    print_compare_feature_results("group-saola", "anova_f_statistic",
                                  extract_same_features(select_features,
                                                        select_top_k_features_anova_f_statistic(data[:, :-1],
                                                                                                data[:, -1],
                                                                                                len(select_features)),)
                                  , feature_map)






if __name__ == "__main__":
    main()
