from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def rf_train_model(data: pd.DataFrame):
    """
    Train a Random Forest Regressor model on the given dataset.

    :param data: pandas DataFrame containing the training data
    :return: Trained model
    """

    # Preprocessing Steps:
    # ... (keep the preprocessing steps as before)

    # Split the data into features and target variable
    X = data.drop('price', axis=1)
    y = data['price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Model Training:
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # Model Evaluation:
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

    return model
