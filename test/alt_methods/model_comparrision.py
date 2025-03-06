from typing import List
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel
import numpy as np
from test.util.get_feature_collections import extract_idx_to_feature


def random_forest_classifier_acc_val(x: np.ndarray, y: np.ndarray, selected_features: List[int]) -> None:
    '''
    Train model on features and selected features and compare how it does both in accuracy
    and in cross validation accuracy
    '''

    x_selected = x[:, np.array(selected_features)]

    # Split data before feature selection
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # Split data after feature selection
    x_train_sel, x_test_sel, _, _ = train_test_split(x_selected, y, test_size=0.2, random_state=42)

    clf_full = RandomForestClassifier(random_state=42)
    clf_full.fit(x_train, y_train)
    y_pred_full = clf_full.predict(x_test)
    acc_full = accuracy_score(y_test, y_pred_full)

    clf_selected = RandomForestClassifier(random_state=42)
    clf_selected.fit(x_train_sel, y_train)
    y_pred_selected = clf_selected.predict(x_test_sel)
    acc_selected = accuracy_score(y_test, y_pred_selected)

    print(f"Accuracy with all features: {acc_full:.4f}")
    print(f"Accuracy with selected features: {acc_selected:.4f}")

    cv_score_full = cross_val_score(clf_full, x, y, cv=5)
    cv_score_selected = cross_val_score(clf_selected, x_selected, y, cv=5)

    print(f"Cross-validation accuracy (All Features): {np.mean(cv_score_full):.4f}")
    print(f"Cross-validation accuracy (Selected Features): {np.mean(cv_score_selected):.4f}")

    t_stat, p_value = ttest_rel(cv_score_full, cv_score_selected)

    print(f"T-test statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

    # Check significance at 5% level
    if p_value < 0.05:
        print("The cross-validation accuracy difference is statistically significant!")
    else:
        print("No significant difference in cross-validation accuracy.")


def pca(x: np.ndarray, y: np.ndarray, k=5) -> None:
    '''
    If selected features still maintain boundaries between classes, selected
    features successful
    '''
    max_features = min(x.shape[1], k)
    x_selected = x[:, :max_features]

    pca_full = PCA(n_components=2)
    x_pca_full = pca_full.fit_transform(x)

    pca_selected = PCA(n_components=2)
    x_pca_selected = pca_selected.fit_transform(x_selected)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].scatter(x_pca_full[:, 0], x_pca_full[:, 1], c=y.astype(int), cmap="viridis", alpha=0.7)
    ax[0].set_title("PCA Projection (All Features)")
    ax[0].set_xlabel("Principal Component 1")
    ax[0].set_ylabel("Principal Component 2")

    ax[1].scatter(x_pca_selected[:, 0], x_pca_selected[:, 1], c=y.astype(int), cmap="viridis", alpha=0.7)
    ax[1].set_title(f"PCA Projection (Top {max_features} Features)")
    ax[1].set_xlabel("Principal Component 1")
    ax[1].set_ylabel("Principal Component 2")

    plt.show()
