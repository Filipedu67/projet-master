"""
This module contains the Lasso Regression model.
"""

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import numpy as np

from models.custom_methods import get_cv_scores, evaluate_model
from preprocess.preprocess import COLUMN_TO_PREDICT

# Define the parameter grid
param_grid = {
    'alpha': [10, 5, 2, 1, 0.1, 0.01, 0.001],
    'max_iter': [1000],
}


def lasso_tune_hyper_parameters(data: pd.DataFrame):
    """
    Train a Lasso regression model on the given dataset.

    :param data: pandas DataFrame containing the training data
    :return: Trained model
    """

    # Encode categorical variables using Label Encoder
    label_encoder = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = label_encoder.fit_transform(data[column])

    # Split the data into features and target variable
    X = data.drop(COLUMN_TO_PREDICT, axis=1)
    y = data[COLUMN_TO_PREDICT]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize the base model
    lasso = Lasso(alpha=1, max_iter=1000)

    # Initialize the Grid Search model
    grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                               verbose=1,
                               n_jobs=-1)

    # Assuming X_train and y_train are your features and target variable
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", -grid_search.best_score_)  # Negate because it's negative mean squared error


def lasso_train_model(data: pd.DataFrame):
    """
    Train a Lasso Regression model on the given dataset.

    :param data: pandas DataFrame containing the training data
    :return: Trained model
    """

    # Encode categorical variables using Label Encoder
    label_encoder = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = label_encoder.fit_transform(data[column])

    # Split the data into features and target variable
    X = data.drop(COLUMN_TO_PREDICT, axis=1)
    y = data[COLUMN_TO_PREDICT]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Model Training:
    model = Lasso()
    model.fit(X_train, y_train)

    get_cv_scores(model, X_test, y_test)

    # Model Evaluation:
    evaluate_model(model, X_test, y_test)

    return model


def lasso_print_model_with_kfold(data: pd.DataFrame, n_splits=30):
    """
    Train a Lasso Regression model using K-Fold Cross-Validation.

    :param data: pandas DataFrame containing the training data
    :param n_splits: Number of splits for K-Fold Cross-Validation
    """

    # Split the data into features and target variable
    X = data.drop(COLUMN_TO_PREDICT, axis=1)
    y = data[COLUMN_TO_PREDICT]

    # Initialize the KFold cross-validator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize lists to store the results for each fold
    mae_scores = []
    mse_scores = []
    rmse_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Model Training:
        model = Lasso()
        model.fit(X_train, y_train)

        # Model Evaluation:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Append the scores
        mae_scores.append(mae)
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

    # Calculate and print the mean of the metrics
    print(f"Mean MAE: {np.mean(mae_scores)}")
    print(f"Mean MSE: {np.mean(mse_scores)}")
    print(f"Mean RMSE: {np.mean(rmse_scores)}")
    print(f"Mean R²: {np.mean(r2_scores)}")
