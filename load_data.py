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


def load_csv_data(file_path, separator=','):
    """
    Loads a CSV file into a pandas DataFrame.

    :param file_path: Path to the CSV file
    :param separator: Separator used in the CSV file
    :return: pandas DataFrame containing the data
    """
    try:
        # Load the CSV file into a DataFrame
        # dtype={18: float, 23: float, 24: float, 26: float, 28: float, 29: float, 31: float, 32: float, 33: float, 41: float}

        df = pd.read_csv(file_path, sep=separator)

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

    if 'No disposition' in df.columns:
        df['No disposition'] = pd.to_numeric(df['No disposition'], errors='coerce')

    if 'Date mutation' in df.columns:
        df['Date mutation'] = df['Date mutation'].astype(str)

    if 'Date mutation' in df.columns:
        df['Date mutation'] = pd.to_datetime(df['Date mutation'], errors='coerce', utc=True)

    if 'Valeur fonciere' in df.columns:
        df['Valeur fonciere'] = df['Valeur fonciere'].replace('[^0-9.-]', '', regex=True)
        df['Valeur fonciere'] = df['Valeur fonciere'].str.replace(',', '.').astype(float)

    if 'Valeur fonciere' in df.columns:
        df['Valeur fonciere'] = pd.to_numeric(df['Valeur fonciere'], errors='coerce')

    if 'No voie' in df.columns:
        df['No voie'] = pd.to_numeric(df['No voie'], errors='coerce')

    if 'B/T/Q' in df.columns:
        df['B/T/Q'] = df['B/T/Q'].astype(str)

    if 'Type de voie' in df.columns:
        df['Type de voie'] = df['Type de voie'].astype(str)

    if 'Code voie' in df.columns:
        df['Code voie'] = df['Code voie'].astype(str)

    if 'Voie' in df.columns:
        df['Voie'] = df['Voie'].astype(str)

    if 'Code postal' in df.columns:
        df['Code postal'] = pd.to_numeric(df['Code postal'], errors='coerce')

    if 'Commune' in df.columns:
        df['Commune'] = df['Commune'].astype(str)

    if 'Code departement' in df.columns:
        df['Code departement'] = pd.to_numeric(df['Code departement'], errors='coerce')

    if 'Code commune' in df.columns:
        df['Code commune'] = pd.to_numeric(df['Code commune'], errors='coerce')

    if 'Section' in df.columns:
        df['Section'] = df['Section'].astype(str)

    if 'No plan' in df.columns:
        df['No plan'] = pd.to_numeric(df['No plan'], errors='coerce')

    if 'No Volume' in df.columns:
        df['No Volume'] = df['No Volume'].astype(str)

    if '1er lot' in df.columns:
        df['1er lot'] = pd.to_numeric(df['1er lot'], errors='coerce')

    if 'Surface Carrez du 1er lot' in df.columns:
        df['Surface Carrez du 1er lot'] = pd.to_numeric(df['Surface Carrez du 1er lot'], errors='coerce')

    if '2eme lot' in df.columns:
        df['2eme lot'] = pd.to_numeric(df['2eme lot'], errors='coerce')

    if 'Surface Carrez du 2eme lot' in df.columns:
        df['Surface Carrez du 2eme lot'] = pd.to_numeric(df['Surface Carrez du 2eme lot'], errors='coerce')

    if '3eme lot' in df.columns:
        df['3eme lot'] = pd.to_numeric(df['3eme lot'], errors='coerce')

    if 'Surface Carrez du 3eme lot' in df.columns:
        df['Surface Carrez du 3eme lot'] = pd.to_numeric(df['Surface Carrez du 3eme lot'], errors='coerce')

    if '4eme lot' in df.columns:
        df['4eme lot'] = pd.to_numeric(df['4eme lot'], errors='coerce')

    if 'Surface Carrez du 4eme lot' in df.columns:
        df['Surface Carrez du 4eme lot'] = pd.to_numeric(df['Surface Carrez du 4eme lot'], errors='coerce')

    if '5eme lot' in df.columns:
        df['5eme lot'] = pd.to_numeric(df['5eme lot'], errors='coerce')

    if 'Surface Carrez du 5eme lot' in df.columns:
        df['Surface Carrez du 5eme lot'] = pd.to_numeric(df['Surface Carrez du 5eme lot'], errors='coerce')

    if 'Nombre de lots' in df.columns:
        df['Nombre de lots'] = pd.to_numeric(df['Nombre de lots'], errors='coerce')

    if 'Code type local' in df.columns:
        df['Code type local'] = pd.to_numeric(df['Code type local'], errors='coerce')

    if 'Type local' in df.columns:
        df['Type local'] = df['Type local'].astype(str)

    if 'Surface reelle bati' in df.columns:
        df['Surface reelle bati'] = pd.to_numeric(df['Surface reelle bati'], errors='coerce')

    if 'Nombre pieces principales' in df.columns:
        df['Nombre pieces principales'] = pd.to_numeric(df['Nombre pieces principales'], errors='coerce')

    if 'Nature culture' in df.columns:
        df['Nature culture'] = df['Nature culture'].astype(str)

    if 'Nature culture speciale' in df.columns:
        df['Nature culture speciale'] = df['Nature culture speciale'].astype(str)

    if 'Surface terrain' in df.columns:
        df['Surface terrain'] = pd.to_numeric(df['Surface terrain'], errors='coerce')

    if 'valeur_fonciere' in df.columns:
        df['valeur_fonciere'] = pd.to_numeric(df['valeur_fonciere'], errors='coerce')

    if 'numero_disposition' in df.columns:
        df['numero_disposition'] = pd.to_numeric(df['numero_disposition'], errors='coerce')

    if 'adresse_numero' in df.columns:
        df['adresse_numero'] = pd.to_numeric(df['adresse_numero'], errors='coerce')

    if 'adresse_code_voie' in df.columns:
        df['adresse_code_voie'] = pd.to_numeric(df['adresse_code_voie'], errors='coerce', downcast='integer')

    if 'code_postal' in df.columns:
        df['code_postal'] = pd.to_numeric(df['code_postal'], errors='coerce')

    if 'code_commune' in df.columns:
        df['code_commune'] = pd.to_numeric(df['code_commune'], errors='coerce', downcast='integer')

    if 'numero_volume' in df.columns:
        df['numero_volume'] = pd.to_numeric(df['numero_volume'], errors='coerce', downcast='integer')

    if 'nombre_lots' in df.columns:
        df['nombre_lots'] = pd.to_numeric(df['nombre_lots'], errors='coerce')

    if 'type_local' in df.columns:
        df['type_local'] = df['type_local'].astype(str)  # Keeping type_local as string

    if 'surface_reelle_bati' in df.columns:
        df['surface_reelle_bati'] = pd.to_numeric(df['surface_reelle_bati'], errors='coerce')

    if 'nombre_pieces_principales' in df.columns:
        df['nombre_pieces_principales'] = pd.to_numeric(df['nombre_pieces_principales'], errors='coerce')

    if 'surface_terrain' in df.columns:
        df['surface_terrain'] = pd.to_numeric(df['surface_terrain'], errors='coerce')

    if 'longitude' in df.columns:
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    if 'latitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')

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
