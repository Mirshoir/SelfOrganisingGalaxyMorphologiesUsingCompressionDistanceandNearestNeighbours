# src/svm_rbf_classify.py

import numpy as np
from typing import Tuple
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# -------------------------------------------------------------
# 1. Train SVM with RBF kernel
# -------------------------------------------------------------
def train_svm_rbf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 10.0,
    gamma: str = "scale",
) -> SVC:

    """
    C = regularization strength (higher = tighter boundary)
    gamma = controls curvature ("scale" is best default)
    """

    model = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        probability=False,
    )

    model.fit(X_train, y_train)
    return model


# -------------------------------------------------------------
# 2. Evaluate SVM classifier
# -------------------------------------------------------------
def evaluate_svm(
    model: SVC,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, np.ndarray, str]:

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return acc, cm, report
