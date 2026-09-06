from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
import json


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression baseline model.

    Returns
    -------
    model
        Trained sklearn Pipeline.
    """

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    print("Logistic Regression training completed.")

    return model

def train_decision_tree(X_train, y_train):
    """
    Train a Decision Tree baseline model.

    Returns
    -------
    model
        Trained Decision Tree classifier.
    """

    model = DecisionTreeClassifier(
        random_state=42,
    )

    model.fit(X_train, y_train)

    print("Decision Tree training completed.")

    return model

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest baseline model.

    Returns
    -------
    model
        Trained Random Forest classifier.
    """

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Random Forest training completed.")

    return model

def train_xgboost(X_train, y_train):
    """
    Train an XGBoost baseline model.

    Returns
    -------
    model
        Trained XGBoost classifier.
    """

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("XGBoost training completed.")

    return model

def tune_logistic_regression(X_train, y_train):
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__solver": ["lbfgs", "liblinear"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="average_precision",
        cv=5,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    print("Logistic Regression tuning completed.")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV average precision: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

def tune_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="average_precision",
        cv=5,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    print("Random Forest tuning completed.")
    print(f"Best parameters: {grid_search.best_params_}")
    print(
        f"Best CV average precision: "
        f"{grid_search.best_score_:.4f}"
    )

    return grid_search.best_estimator_

def train_selected_random_forest(X_train, y_train):
    """
    Train Random Forest using the hyperparameters
    selected during cross-validation tuning.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Selected Random Forest training completed.")

    return model

def save_model(
    model,
    model_path,
):
    """
    Save a trained model as a reusable artifact.
    """

    model_path = Path(model_path)

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path,
    )

    print(f"Model saved successfully: {model_path}")

def save_model_metadata(
    metadata,
    metadata_path,
):
    """
    Save model metadata required for reproducible inference.
    """

    metadata_path = Path(metadata_path)

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(f"Model metadata saved successfully: {metadata_path}")
