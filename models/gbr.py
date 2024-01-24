import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np


def train_model(data: pd.DataFrame):
    """
    Train a Gradient Boosting Regressor model on the given dataset.

    :param data: pandas DataFrame containing the training data
    :return: Trained model
    """

    # Preprocessing Steps:

    # 3. Encode categorical variables using Label Encoder
    label_encoder = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = label_encoder.fit_transform(data[column])

    # Split the data into features and target variable
    X = data.drop('price', axis=1)
    y = data['price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training:
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # print the score of the model
    print(f"Score: {model.score(X_test, y_test)}")

    return model


def predict_price(model, input_attributes):
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
