import os

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
import numpy as np


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
