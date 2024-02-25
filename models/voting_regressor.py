import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

from preprocess.preprocess import COLUMN_TO_PREDICT


def voting_regressor_train_model(data: pd.DataFrame):
    """
    Train a Voting Regressor model combining Gradient Boosting and Random Forest on the given dataset.

    :param data: pandas DataFrame containing the training data
    :return: Trained Voting Regressor model
    """

    # Encode categorical variables using Label Encoder
    label_encoder = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = label_encoder.fit_transform(data[column])

    # Split the data into features and target variable
    X = data.drop(COLUMN_TO_PREDICT, axis=1)
    y = data[COLUMN_TO_PREDICT]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize the models
    gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=3)
    rf = RandomForestRegressor(n_estimators=300, max_depth=3)

    # Initialize the Voting Regressor
    voting_regressor = VotingRegressor(estimators=[('gbr', gbr), ('rf', rf)])

    # Train the Voting Regressor
    voting_regressor.fit(X_train, y_train)

    # Evaluate the model
    y_pred = voting_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the metrics
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R² score: {r2}")

    return voting_regressor
