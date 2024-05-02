"""
This module contains custom methods for evaluating models.
"""

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

from models.custom_methods import get_cv_scores, evaluate_model
from preprocess.preprocess import COLUMN_TO_PREDICT


def knn_train_model(data: pd.DataFrame, k=10):
    """
    Train a K-Nearest Neighbors model on the given dataset.
    :param data:    pandas DataFrame containing the training data
    :param k:       Number of neighbors to consider
    :return:        Trained model
    """
    label_encoder = LabelEncoder()

    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = label_encoder.fit_transform(data[column])

    X = data.drop(COLUMN_TO_PREDICT, axis=1)
    y = data[COLUMN_TO_PREDICT].apply(lambda x: 10000 * round(x / 10000))
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    get_cv_scores(model, X_test, y_test)

    # Model Evaluation:
    evaluate_model(model, X_test, y_test)

    return model


def knn_print_model_with_kfold(data: pd.DataFrame, n_splits=30, k=5):
    """
    Train a Neural Network model using K-Fold Cross-Validation.

    :param data: pandas DataFrame containing the training data
    :param n_splits: Number of splits for K-Fold Cross-Validation
    """

    # Split the data into features and target variable
    X = data.drop(COLUMN_TO_PREDICT, axis=1)
    y = data[COLUMN_TO_PREDICT].round(-4)

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
        model = KNeighborsClassifier(n_neighbors=k)
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


def knn_predict_price(model, input_attributes):
    """
    Predict the house price using the trained model and input attributes.

    :param model: Trained model
    :param input_attributes: Dictionary with input features
    :return: Predicted price
    """
    # Create a DataFrame from the input attributes
    input_data = pd.DataFrame([input_attributes])

    # Ensure the column order matches the training data
    input_data = input_data[model.feature_names_in_]

    # Predict and return the house price
    predicted_price = model.predict(input_data)
    return predicted_price[0]  # Return a single value
