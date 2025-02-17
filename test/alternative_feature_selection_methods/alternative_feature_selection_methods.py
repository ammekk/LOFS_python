from tokenize import String
from typing import Union, Dict, List

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from test.util.get_feature_collections import extract_idx_to_feature


def select_top_k_features_anova_f_statistic(x: np.ndarray, y: np.ndarray, feature_map: dict[int,
any], k: int = 5) -> List[int]:
    # feature selection method that selects the top k features based on their statistical significance
    # with respect to the target variable, typically using the ANOVA F-statistic for classification.
    selector = SelectKBest(score_func=f_classif, k=k)  # Select top 5 features
    X_new = selector.fit_transform(x, y)

    # Get selected feature indices
    selected_features = selector.get_support(indices=True)

    selected_features = selected_features.tolist() if isinstance(selected_features, np.ndarray) \
        else [0, 0, 0, 0]
    selected_named_features = extract_idx_to_feature(feature_map,selected_features)
    print(f"Selected features (ANOVA F-statistic): {selected_named_features}")
    return selected_features


def select_top_k_features_mutual_information(x: np.ndarray, y: np.ndarray,  feature_map: dict[int,
any], k: int = 5) -> List[int]:
    # uses the mutual information criterion to evaluate the dependency between features
    # and the target variable, selecting the top k features.

    selector = SelectKBest(score_func=mutual_info_classif, k=5)
    _ = selector.fit_transform(x, y)

    selected_features = selector.get_support(indices=True)
    selected_features = selected_features.tolist() if isinstance(selected_features, np.ndarray) \
        else [0, 0, 0, 0]
    selected_named_features = extract_idx_to_feature(feature_map,selected_features)
    print(f"Selected features (Mutual Information): {selected_named_features}")
    return selected_features


def random_forest_classifier_select_features(x: np.ndarray, y: np.ndarray, feature_map: dict[int, any],
                                             n_estimators=100, k=5) -> List[int]:
    # trains a Random Forest classifier on the input features (x) and target (y),
    # then selects the top k features based on their importance scores.
    rf = RandomForestClassifier(n_estimators=n_estimators)
    rf.fit(x, y)

    feature_importances = rf.feature_importances_

    k_indices = np.argsort(feature_importances)[-k:]  # Indices of top 5 features
    selected_named_features = extract_idx_to_feature(feature_map, k_indices)

    print(f"Selected feature indices: {selected_named_features}")
    return k_indices.tolist()
