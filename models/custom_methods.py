"""
This module contains custom methods for evaluating models.
"""

import os

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

all_messages = []

def get_cv_scores(clf, X_test, y_test):
    """
    Get cross-validation scores for a model.
    :param clf:     The model to evaluate.
    :param X_test:  The features to evaluate.
    :param y_test:  The target variable to evaluate.
    :return:        The cross-validation scores.
    """
    kf = KFold(n_splits=30, shuffle=True, random_state=None)  # Change random_state for different shuffles

    cv_scores = cross_val_score(clf, X_test, y_test, cv=kf)

    log(f"CV Scores: {cv_scores}")
    log(f"Mean accuracy: {np.mean(cv_scores)}")
    log(f"Standard deviation: {np.std(cv_scores)}")

    return cv_scores


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model using common regression metrics.
    :param model:   The model to evaluate.
    :param X_test:  The features to evaluate.
    :param y_test:  The target variable to evaluate.
    :return:        None
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the metrics
    log(f"Score (R²): {model.score(X_test, y_test)}")
    log(f"Mean Absolute Error (MAE): {mae}")
    log(f"Mean Squared Error (MSE): {mse}")
    log(f"Root Mean Squared Error (RMSE): {rmse}")
    log(f"R² score: {r2}")


def calculate_threshold_accuracy(y_true, y_pred, threshold=20000):
    """
    Calculates the percentage of predictions that fall within a specified threshold of the actual values.

    Parameters:
    - y_true: The actual values.
    - y_pred: The predicted values.
    - threshold: The threshold for considering a prediction as accurate.

    Returns:
    - accuracy: The percentage of predictions that are considered accurate.
    """
    # Calculate the absolute difference between actual and predicted values
    abs_diff = np.abs(y_true - y_pred)

    # Determine which differences are within the threshold
    within_threshold = abs_diff <= threshold

    # Calculate the accuracy as the percentage of predictions within the threshold
    accuracy = np.mean(within_threshold) * 100  # Convert to percentage

    return accuracy


def plot_accuracy(cv_scores, model_name, file_name, source_language):
    """
    Plot the cross-validation accuracies for a model.
    :param cv_scores:           The cross-validation scores.
    :param model_name:          The name of the model.
    :param file_name:           The name of the file.
    :param source_language:     The source language of the data.
    :return:                    None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cv_scores, marker='o', linestyle='-', color='b')
    plt.title(f'Cross-Validation Accuracies for {model_name} on {file_name} for {source_language}')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # create cv_accuracies folder if it doesn't exist in the current directory
    if not os.path.exists('cv_accuracies'):
        os.makedirs('cv_accuracies')

    plt.savefig(f'cv_accuracies/{file_name}_{model_name}_{source_language}.png')
    plt.show()


def log(message):
    """
    Log a message.
    :param message:     Message to log
    :return:            None
    """
    all_messages.append(message)
    print(message)
