import pandas as pd

import shap
from shap import Explanation

from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from xgboost import XGBClassifier

OVERSAMPLING = True


def prepare_train_test_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """This function splits the feature matrix into training and test sets and applies
    oversampling to the training set to allow for better learning of the minority class.

    Args:
        df: Cleaned and pre-processed feature matrix.

    Returns:
        pd.DataFrame: Training set.
        pd.Series: Training labels.
        pd.DataFrame: Test set.
        pd.Series: Test labels.

    """

    # Define features for the model
    feature_cols = [col for col in df.columns if col not in ['instance_weight', "target", "is_train"]]

    # Train and test split
    X_train, y_train = df[df["is_train"]==1][feature_cols].copy(), df[df["is_train"]==1]["target"].copy()
    X_test, y_test = df[df["is_train"]==0][feature_cols].copy(), df[df["is_train"]==0]["target"].copy()

    if OVERSAMPLING:
        # Define oversampling object and create new (balanced) training sets
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)

    print(f" - Training set shape: {X_train.shape}")
    print(f" - Training labels shape: {y_train.shape}")
    print(f" - Test set shape: {X_test.shape}")
    print(f" - Test labels shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> XGBClassifier:
    """This function defines a classification model to learn associations between
    features and income levels. Additionally, it prints out a number of validation
    performance metrics like the confusion matrix, a standard classification report
    with precision, recall, and F1 scores, and the ROC value.

    Args:
        X_train: Training set.
        y_train: Training labels.
        X_test: Test set.
        y_test: Test labels.

    Returns:
        XGBClassifier: The trained XGBoost classifier model.

    """

    # Define XGBoost Classifier with pre-defined parameters, which have
    # been investigated in the notebook.
    xgb_model = XGBClassifier(n_estimators=500, max_depth=9, n_jobs=4, random_state=42)
    
    # Train model
    print(" - Training XGB model with Random OverSampling")
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Assess the classification performance by printing out standard
    # metrics
    _compute_validation_metrics(xgb_model, X_test, y_test)

    return xgb_model


def _compute_validation_metrics(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """This function prints out a number of validation performance metrics
    like the confusion matrix, a standard classification report with precision,
    recall, and F1 scores, and the ROC value.

    Args:
        model: The trained XGBoost classifier model.
        X_test: Test set.
        y_test: Test labels.

    """

    # Predict class labels
    y_pred = model.predict(X_test)
    # Predict probabilities for the positive class (assuming a binary classifier)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Compute and print the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(" - Confusion Matrix:")
    print(cm)

    # Compute and print the classification report (precision, recall, F1 scores)
    report = classification_report(y_test, y_pred)
    print("\n - Classification Report:")
    print(report)

    # Compute and print the ROC AUC score
    roc_value = roc_auc_score(y_test, y_pred_proba)
    print("\n - ROC AUC Score:")
    print(roc_value)


def explainability_engine(model: XGBClassifier, X_test: pd.DataFrame) -> tuple[dict, Explanation]:
    """This function combines two different methods to explain the trained model,
    in order to understand which are the key drivers for model predictions, and
    understand how the different feature values influence the outcome.

    Args:
        model: The trained XGBoost classifier model.
        X_test: Test set.

    Returns:
        tuple[dict, Explanation]: Both a dictionary with feature importances based on
        split information gains and the SHAP explanation values.

    """

    # Get feature importances (built-in method)
    print(" - Computing feature importance based on split information gain.")
    importance_dict = model.get_booster().get_score(importance_type='gain')
    importance_dict = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)

    # Get SHAP explanations
    print(" - Computing SHAP explanations (may take up to 5-10 minutes).")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    return importance_dict, shap_values


