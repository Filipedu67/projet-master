

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def train_model(data: pd.DataFrame):
    """
    Train a Random Forest Regressor model on the given dataset.

    :param data: pandas DataFrame containing the training data
    :return: Trained model
    """

    # Split the data into features and target variable
    X = data.drop('price', axis=1)
    y = data['price']

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    # Model Training:
    classifier = SVC()

    parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

    grid_search = GridSearchCV(classifier, parameters)

    grid_search.fit(x_train, y_train)
    
    best_estimator = grid_search.best_estimator_
    
    # Model Evaluation:
    y_prediction = best_estimator.predict(x_test)

    acc_score = accuracy_score(y_prediction, y_test)
    mse = mean_squared_error(y_test, y_prediction)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_prediction)
    r2 = r2_score(y_test, y_prediction)

    # Print the metrics
    print(f"Score (R²): {best_estimator.score(x_test, y_test)}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R² score: {r2}")
    print(f"Accuracy score: {acc_score }")

    return best_estimator
