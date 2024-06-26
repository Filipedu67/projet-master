"""
This script calculates the mean of the price per square meter for each city and evaluates the model on the test set.
"""

from load_data import load_json_data
import math, sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocess.preprocess import COLUMN_TO_PREDICT

cities = [
    'bordeaux',
    'lille',
    'lyon',
    'marseille',
    'montpellier',
    'nantes',
    'nice',
    'paris',
    'strasbourg',
    'toulouse'
]


def get_city_dataframes():
    """
    Load the data for each city and return a dictionary of dataframes.
    :return:    Dictionary of dataframes
    """
    dfs = {}
    for city in cities:
        data = load_json_data(f"data/data-{city}.json")
        dfs[city] = data
    return dfs


def evaluate_mean(city_df) -> None:
    """
    Evaluate the mean model on the test set.
    :param city_df:     pandas DataFrame containing the data for a city
    :return:            None
    """
    X, y = city_df['surface'], city_df[COLUMN_TO_PREDICT]
    # seperating the data into 80/20 train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # calculation the mean on the train set
    mean = (y_train / X_train).mean()
    y_pred = X_test * mean  # predicting the prices

    # metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"mae: {mae}")
    print(f"mse: {mse}")
    print(f"sqrt(mse): {math.sqrt(mse)}")
    print(f"r2: {r2}")


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print(f'Usage: python3 {sys.argv[0]} <city_name>')
        sys.exit(1)

    city = sys.argv[1]

    if not city in cities:
        print(f"City not supported, please use one of the following: {', '.join(cities)}")
        sys.exit(1)

    means = get_city_dataframes()
    evaluate_mean(means[city])
