"""
This module contains the function to predict the house price using the trained model.
"""

import pandas as pd


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