from load_data import load_data
from typing import Dict
import math 

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    dfs = {} 
    for city in cities:
        data = load_data(f"data/cities/data-{city}.json")
        dfs[city] = data
    return dfs

def evaluate_mean(city_df) -> None:
    X, y = city_df['surface'], city_df['price']
    # seperating the data into 80/20 train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
    
    # calculation the mean on the train set
    mean = (y_train / X_train).mean()
    y_pred = X_test * mean # predicting the prices

    # metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"mae: {mae}")
    print(f"mse: {mse}")
    print(f"sqrt(mse): {math.sqrt(mse)}")
    print(f"r2: {r2}")


if __name__ == '__main__':
    means = get_city_dataframes()
    evaluate_mean(means['strasbourg'])