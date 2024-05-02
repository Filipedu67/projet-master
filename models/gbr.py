"""
This module contains the functions to train and evaluate a Gradient Boosting Regressor model on the given dataset.
"""

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import numpy as np

from models.custom_methods import get_cv_scores, evaluate_model
from preprocess.preprocess import COLUMN_TO_PREDICT

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}


def gbr_tune_hyper_parameters(data: pd.DataFrame):
    """
    Train a Gradient Boosting Regressor model on the given dataset.

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
    gbr = GradientBoostingRegressor()

    # Initialize the Grid Search model
    grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1,
                               n_jobs=-1)

    # Assuming X_train and y_train are your features and target variable
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", -grid_search.best_score_)  # Negate because it's negative mean squared error


def gbr_train_model(data: pd.DataFrame):
    """
    Train a Gradient Boosting Regressor model on the given dataset.

    :param data: pandas DataFrame containing the training data
    :return: Trained model
    """

    # Encode categorical variables using Label Encoder
    label_encoder = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data.loc[:, column] = label_encoder.fit_transform(data[column])

    # Split the data into features and target variable
    X = data.drop(COLUMN_TO_PREDICT, axis=1)
    y = data[COLUMN_TO_PREDICT]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Model Training:
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    get_cv_scores(model, X_test, y_test)

    # Model Evaluation:
    evaluate_model(model, X_test, y_test)

    return model


def gbr_print_model_with_kfold(data: pd.DataFrame, n_splits=30):
    """
    Train a Gradient Boosting Regressor model using K-Fold Cross-Validation.

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
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.2, max_depth=4)
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
