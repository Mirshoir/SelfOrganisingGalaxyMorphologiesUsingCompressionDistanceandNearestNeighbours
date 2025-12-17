# src/knn_classify.py

import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -------------------------------------------------------------
# 1. Split + scale features (consistent for KNN/SVM)
# -------------------------------------------------------------
def split_and_scale(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# -------------------------------------------------------------
# 2. Train KNN classifier
# -------------------------------------------------------------
def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int = 5,
) -> KNeighborsClassifier:

    model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    model.fit(X_train, y_train)
    return model


# -------------------------------------------------------------
# 3. Evaluate KNN classifier
# -------------------------------------------------------------
def evaluate_knn(
    model: KNeighborsClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, np.ndarray, str]:

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return acc, cm, report
