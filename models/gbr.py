import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import numpy as np


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
            data[column] = label_encoder.fit_transform(data[column])

    # Split the data into features and target variable
    X = data.drop('price', axis=1)
    y = data['price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Model Training:
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
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


def gbr_print_model_with_kfold(data: pd.DataFrame, n_splits=30):
    """
    Train a Gradient Boosting Regressor model using K-Fold Cross-Validation.

    :param data: pandas DataFrame containing the training data
    :param n_splits: Number of splits for K-Fold Cross-Validation
    """

    # Split the data into features and target variable
    X = data.drop('price', axis=1)
    y = data['price']

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
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
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



def general_predict_price(model, input_attributes):
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
