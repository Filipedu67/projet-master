import json

import pandas
import pandas as pd


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

        df = pd.read_csv(file_path, sep='|')

        df = correct_data_types_v2(df)

        return df

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def correct_data_types_v2(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Correct the data types of the given DataFrame.

    :param df: pandas DataFrame
    :return: pandas DataFrame with corrected data types
    """
    df['No disposition'] = pd.to_numeric(df['No disposition'], errors='coerce')

    df['Date mutation'] = df['Date mutation'].astype(str)

    df['Date mutation'] = pd.to_datetime(df['Date mutation'], errors='coerce', utc=True)

    if 'Valeur fonciere' in df.columns:
        df['Valeur fonciere'] = df['Valeur fonciere'].replace('[^0-9.-]', '', regex=True)
        df['Valeur fonciere'] = df['Valeur fonciere'].str.replace(',', '.').astype(float)

    df['Valeur fonciere'] = pd.to_numeric(df['Valeur fonciere'], errors='coerce')

    df['No voie'] = pd.to_numeric(df['No voie'], errors='coerce')

    df['B/T/Q'] = df['B/T/Q'].astype(str)

    df['Type de voie'] = df['Type de voie'].astype(str)

    df['Code voie'] = df['Code voie'].astype(str)

    df['Voie'] = df['Voie'].astype(str)

    df['Code postal'] = pd.to_numeric(df['Code postal'], errors='coerce')

    df['Commune'] = df['Commune'].astype(str)

    df['Code departement'] = pd.to_numeric(df['Code departement'], errors='coerce')

    df['Code commune'] = pd.to_numeric(df['Code commune'], errors='coerce')

    df['Section'] = df['Section'].astype(str)

    df['No plan'] = pd.to_numeric(df['No plan'], errors='coerce')

    # NOTE: There is a type mismatch here...
    # The column is float but there are string values
    df['No Volume'] = df['No Volume'].astype(str)

    # NOTE: There is a type mismatch here...
    # The column is float but there are string values
    df['1er lot'] = pd.to_numeric(df['1er lot'], errors='coerce')

    df['Surface Carrez du 1er lot'] = pd.to_numeric(df['Surface Carrez du 1er lot'], errors='coerce')

    # NOTE: There is a type mismatch here...
    # The column is float but there are string values
    df['2eme lot'] = pd.to_numeric(df['2eme lot'], errors='coerce')

    df['Surface Carrez du 2eme lot'] = pd.to_numeric(df['Surface Carrez du 2eme lot'], errors='coerce')

    # NOTE: There is a type mismatch here...
    # The column is float but there are string values
    df['3eme lot'] = pd.to_numeric(df['3eme lot'], errors='coerce')

    df['Surface Carrez du 3eme lot'] = pd.to_numeric(df['Surface Carrez du 3eme lot'], errors='coerce')

    df['4eme lot'] = pd.to_numeric(df['4eme lot'], errors='coerce')

    df['Surface Carrez du 4eme lot'] = pd.to_numeric(df['Surface Carrez du 4eme lot'], errors='coerce')

    # NOTE: There is a type mismatch here...
    # The column is float but there are string values
    df['5eme lot'] = pd.to_numeric(df['5eme lot'], errors='coerce')

    df['Surface Carrez du 5eme lot'] = pd.to_numeric(df['Surface Carrez du 5eme lot'], errors='coerce')

    df['Nombre de lots'] = pd.to_numeric(df['Nombre de lots'], errors='coerce')

    df['Code type local'] = pd.to_numeric(df['Code type local'], errors='coerce')

    df['Type local'] = df['Type local'].astype(str)

    df['Surface reelle bati'] = pd.to_numeric(df['Surface reelle bati'], errors='coerce')

    df['Nombre pieces principales'] = pd.to_numeric(df['Nombre pieces principales'], errors='coerce')

    df['Nature culture'] = df['Nature culture'].astype(str)

    df['Nature culture speciale'] = df['Nature culture speciale'].astype(str)

    df['Surface terrain'] = pd.to_numeric(df['Surface terrain'], errors='coerce')

    return df


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
