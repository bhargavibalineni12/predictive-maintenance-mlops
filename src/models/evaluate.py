from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import (
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
import numpy as np


def evaluate_model_cv(model, X_train, y_train, cv=5):
    scoring = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]

    results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
    )

    metrics = {
        "accuracy": results["test_accuracy"].mean(),
        "precision": results["test_precision"].mean(),
        "recall": results["test_recall"].mean(),
        "f1": results["test_f1"].mean(),
        "roc_auc": results["test_roc_auc"].mean(),
        "average_precision": results["test_average_precision"].mean(),
    }

    return metrics


def find_best_threshold_cv(
    model,
    X_train,
    y_train,
    cv=5,
):
    """
    Find the best classification threshold using
    out-of-fold predictions from the training data.

    The final test set is not used.
    """

    probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_train,
        probabilities,
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    best_index = np.argmax(f1_scores)

    best_threshold = thresholds[best_index]
    best_precision = precision[best_index]
    best_recall = recall[best_index]
    best_f1 = f1_scores[best_index]

    print("\nBest threshold selected using training CV predictions:")
    print(f"Threshold: {best_threshold:.4f}")
    print(f"Precision: {best_precision:.4f}")
    print(f"Recall: {best_recall:.4f}")
    print(f"F1: {best_f1:.4f}")

    return best_threshold


def evaluate_final_test(
    model,
    X_test,
    y_test,
    threshold,
):
    """
    Evaluate the locked model and threshold once
    on the untouched final test set.
    """

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "average_precision": average_precision_score(
            y_test,
            probabilities,
        ),
    }

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print("\nFinal Test Set Evaluation:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    return metrics, cm