import pandas as pd


def load_data(file_path):
    # Read JSON file into a DataFrame
    return pd.read_json(file_path)
