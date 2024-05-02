"""
This module contains functions to train a Neural Network model on the given dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from preprocess.preprocess import COLUMN_TO_PREDICT


def nn_train_model(data: pd.DataFrame):
    """
    Train a Neural Network model on the given dataset.

    :param data: pandas DataFrame containing the training data
    :return: Trained model
    """
    # Séparation des données en features (X) et target (y)

    label_encoder = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = label_encoder.fit_transform(data[column])

    X = data.drop(COLUMN_TO_PREDICT, axis=1)
    y = data[COLUMN_TO_PREDICT]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Création du modèle séquentiel
    model = Sequential()

    # Ajout de la couche d'entrée et d'une couche dense
    # Ajout de la couche d'entrée et des couches cachées
    model.add(Dense(units=4096, activation='selu', input_dim=X.shape[1]))
    model.add(Dense(units=2048, activation='selu'))
    model.add(Dense(units=1024, activation='selu'))
    model.add(Dense(units=512, activation='selu'))
    model.add(Dense(units=256, activation='selu'))
    model.add(Dense(units=128, activation='selu'))
    model.add(Dense(units=64, activation='selu'))
    model.add(Dense(units=32, activation='selu'))
    model.add(Dense(units=1, activation='linear'))

    # Compilation du modèle
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Entraînement du modèle
    model.fit(X_train, y_train, epochs=50, verbose=1)

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)

    y_pred = np.nan_to_num(y_pred, nan=0)

    # Check for NaN values in y_test
    nan_indices = np.isnan(y_test)

    # Remove corresponding elements from both y_test and y_pred
    y_test_no_nan = y_test[~nan_indices]
    y_pred_no_nan = y_pred[~nan_indices]

    # Évaluation du modèle
    mse = mean_squared_error(y_test_no_nan, y_pred_no_nan)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_no_nan, y_pred_no_nan)
    r2 = r2_score(y_test_no_nan, y_pred_no_nan)

    # print(f"Score (R²): {model.score(X_test, y_test)}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R² score: {r2}")


def nn_print_model_with_kfold(data: pd.DataFrame, n_splits=30):
    """
    Train a Neural Network model using K-Fold Cross-Validation.

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
        model = Sequential()
        model.add(Dense(units=128, input_dim=X.shape[1], activation='relu'))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=50, verbose=1)

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


def nn_predict_price(model, input_attributes):
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
