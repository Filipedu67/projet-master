from load_data import load_data
import sys, math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

def get_metrics(city_df) -> None:
    # Séparation des données en features (X) et target (y)
    selected_columns = ['surface', 'room', 'bedroom', 'floor', 'elevator', 'furnished', 'propertyType']

    X = city_df[selected_columns]
    y = city_df['price']
    
    X['elevator'] = X['elevator'].fillna(False).astype(int)
    X['furnished'] = X['furnished'].fillna(False).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Création du modèle séquentiel
    model = Sequential()

    # Ajout de la couche d'entrée et d'une couche dense
    # Ajout de la couche d'entrée et des couches cachées
    model.add(Dense(units=128, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1, activation='linear'))


    # Compilation du modèle
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Entraînement du modèle
    model.fit(X_train_scaled, y_train, epochs=50, verbose=1)

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test_scaled)

    y_pred = np.nan_to_num(y_pred, nan=0)

    # Check for NaN values in y_test
    nan_indices = np.isnan(y_test)

    # Remove corresponding elements from both y_test and y_pred
    y_test_no_nan = y_test[~nan_indices]
    y_pred_no_nan = y_pred[~nan_indices]

    # Évaluation du modèle
    mae = mean_absolute_error(y_test_no_nan, y_pred_no_nan)
    mse = mean_squared_error(y_test_no_nan, y_pred_no_nan)
    r2 = r2_score(y_test_no_nan, y_pred_no_nan)
    
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
    get_metrics(means[city])