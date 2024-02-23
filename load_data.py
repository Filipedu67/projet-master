import json

import pandas
import pandas as pd

from preprocess.preprocess import COLUMN_TO_PREDICT

def load_json_data(file_path):
    """
    Loads a JSON file into a pandas DataFrame.

    :param file_path: Path to the JSON file
    :return: pandas DataFrame containing the data
    """

    try:
        # Open the file for reading
        with open(file_path, 'r', encoding="utf8") as file:
            # Parse the JSON data and convert it into a Python object
            data = json.load(file)
            # Convert the JSON data to a DataFrame
            # If the JSON data is an array of objects
            df = pd.DataFrame(data)

            # If the JSON data is nested, you might need to normalize it:
            df = pd.json_normalize(data)

            # correct the date object type
            df = correct_data_types(df)

            return df

    except json.JSONDecodeError as err:
        print(f"Error parsing JSON: {err}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_csv_data(file_path):
    """
    Loads a CSV file into a pandas DataFrame.

    :param file_path: Path to the CSV file
    :return: pandas DataFrame containing the data
    """
    try:
        # Load the CSV file into a DataFrame
        # dtype={18: float, 23: float, 24: float, 26: float, 28: float, 29: float, 31: float, 32: float, 33: float, 41: float}

        df = pd.read_csv(file_path, sep='|', low_memory=False)

        return df

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def correct_data_types(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Correct the data types of the given DataFrame.

    :param df: pandas DataFrame
    :return: pandas DataFrame with corrected data types
    """
    df['elevator'] = df['elevator'].astype(bool)

    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    df['bedroom'] = pd.to_numeric(df['bedroom'], errors='coerce')

    df['description'] = df['description'].astype(str)

    df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce', utc=True)

    df['expired'] = df['expired'].astype(bool)

    df['floor'] = pd.to_numeric(df['floor'], errors='coerce')

    df['furnished'] = df['furnished'].astype(bool)

    df['pricePerMeter'] = pd.to_numeric(df['pricePerMeter'], errors='coerce')

    df['room'] = pd.to_numeric(df['room'], errors='coerce')

    df['surface'] = pd.to_numeric(df['surface'], errors='coerce')

    df['title'] = df['title'].astype(str)

    df['city.department.code'] = pd.to_numeric(df['city.department.code'], errors='coerce')

    df['city.department.name'] = df['city.department.name'].astype(str)

    df['city.location.lat'] = pd.to_numeric(df['city.location.lat'], errors='coerce')

    df['city.location.lon'] = pd.to_numeric(df['city.location.lon'], errors='coerce')

    df['city.name'] = df['city.name'].astype(str)

    df['city.zipcode'] = pd.to_numeric(df['city.zipcode'], errors='coerce')

    return df
