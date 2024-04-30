import os

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
import numpy as np


# def get_cv_scores(model, X, y, n_splits=30, threshold=20000):
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)
#
#     threshold_accuracies = []
#
#     for train_index, test_index in kf.split(X):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#
#         # Clone the original model to ensure each fold starts with an untrained model
#         cloned_model = clone(model)
#         cloned_model.fit(X_train, y_train)
#         y_pred = cloned_model.predict(X_test)
#
#         # Calculate threshold accuracy
#         accuracy = calculate_threshold_accuracy(y_test, y_pred, threshold)
#         threshold_accuracies.append(accuracy)
#
#     # Print the results
#     print(f"CV Threshold-based Accuracies: {threshold_accuracies}")
#     print(f"Mean Threshold-based Accuracy: {np.mean(threshold_accuracies)}")
#     print(f"Standard Deviation: {np.std(threshold_accuracies)}")


def get_cv_scores(clf, X_test, y_test):
    kf = KFold(n_splits=30, shuffle=True, random_state=None)  # Change random_state for different shuffles

    cv_scores = cross_val_score(clf, X_test, y_test, cv=kf)

    print(f"CV Scores: {cv_scores}")
    print(f"Mean accuracy: {np.mean(cv_scores)}")
    print(f"Standard deviation: {np.std(cv_scores)}")

    return cv_scores


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the metrics
    print(f"Score (R²): {model.score(X_test, y_test)}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R² score: {r2}")

# def evaluate_model(model, X_test, y_test, threshold=20000):
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     threshold_accuracy = calculate_threshold_accuracy(y_test, y_pred, threshold)
#
#     # Print the metrics
#     print(f"Score (R²): {model.score(X_test, y_test)}")
#     print(f"Mean Absolute Error (MAE): {mae}")
#     print(f"Mean Squared Error (MSE): {mse}")
#     print(f"Root Mean Squared Error (RMSE): {rmse}")
#     print(f"R² score: {r2}")
#     print(f"Threshold-based Accuracy (±{threshold}): {threshold_accuracy}%")


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
