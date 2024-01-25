import pandas
import pandas as pd
from math import radians, cos, sin, asin, sqrt

# Define the bounds for Paris (Ile de france) (approximate values)
PARIS_LAT_MIN = 48.851981
PARIS_LAT_MAX = 48.997016
PARIS_LON_MIN = 1.682391
PARIS_LON_MAX = 2.532633

# Define the bounds for Strasbourg (approximate values)
STRA_LAT_MIN = 48.497858
STRA_LAT_MAX = 48.661190
STRA_LON_MIN = 7.591532
STRA_LON_MAX = 7.805429

# Define the bounds for lyon (approximate values)
LYON_LAT_MIN = 45.291841
LYON_LAT_MAX = 46.257407
LYON_LON_MIN = 4.259844
LYON_LON_MAX = 5.331176

# Define the bounds for Nantes (approximate values)
NANTES_LAT_MIN = 47.158624
NANTES_LAT_MAX = 47.311649
NANTES_LON_MIN = -1.659836
NANTES_LON_MAX = -1.443447

# Define the bounds for Bordeaux (approximate values)
BORD_LAT_MIN = 44.770868
BORD_LAT_MAX = 44.948455
BORD_LON_MIN = -0.699015
BORD_LON_MAX = -0.467902

# Define the columns you want to keep
COLUMNS_TO_KEEP = ['price', 'elevator', 'location.lat', 'location.lon',
                   'surface', 'bedroom', 'floor',
                   'furnished', 'room']  # Add other column names as needed


def preprocess_data(df: pandas.DataFrame, city: str) -> pandas.DataFrame:
    if city == "paris":
        lat_min = PARIS_LAT_MIN
        lon_min = PARIS_LON_MIN
        lat_max = PARIS_LAT_MAX
        lon_max = PARIS_LON_MAX
    elif city == "strasbourg":
        lon_min = STRA_LON_MIN
        lon_max = STRA_LON_MAX
        lat_min = STRA_LAT_MIN
        lat_max = STRA_LAT_MAX
    elif city == "lyon":
        lat_min = LYON_LAT_MIN
        lon_min = LYON_LON_MIN
        lat_max = LYON_LAT_MAX
        lon_max = LYON_LON_MAX
    elif city == "nantes":
        lat_min = NANTES_LAT_MIN
        lon_min = NANTES_LON_MIN
        lat_max = NANTES_LAT_MAX
        lon_max = NANTES_LON_MAX
    elif city == "bordeaux":
        lat_min = BORD_LAT_MIN
        lon_min = BORD_LON_MIN
        lat_max = BORD_LAT_MAX
        lon_max = BORD_LON_MAX
    else:
        # TODO: Add more cities
        print("Invalid city")
        return None

    # Starting data clean up

    df = convert_bool_to_int(df)

    df = filter_columns(df, COLUMNS_TO_KEEP)

    df = clean_data_with_repeated_longitudes(df)

    df = clean_location_data(df, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)

    df = handle_missing_values(df)

    df = add_distance_features(df, city)

    print_rows_with_nulls(df)

    # end of cleaning up the data

    return df


def clean_data_with_repeated_longitudes(df, column='location.lon', threshold=20):
    """
    Remove rows from the DataFrame where the longitude value appears more than 'threshold' times.

    :param df: pandas DataFrame containing the data.
    :param column: The name of the column to check for repeated longitude values.
    :param threshold: The number of times the longitude value is allowed to repeat.
    :return: Cleaned pandas DataFrame.
    """

    # Count the occurrences of each longitude value
    lon_counts = df[column].value_counts()

    # Find the longitude values that occur more than 'threshold' times
    lon_to_remove = lon_counts[lon_counts > threshold].index

    # Remove rows with these longitude values
    df_cleaned = df[~df[column].isin(lon_to_remove)]

    return df_cleaned


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def get_important_places(city):
    if city == 'paris':
        return {
            'parisGareEst': (48.8763, 2.359),
            'parisGareMontparnasse': (48.8409, 2.321),
            'parisEiffel': (48.8584, 2.2945),
            'louvreMuseum': (48.8606, 2.3376),
            'notreDameCathedral': (48.8529, 2.3500),
            'sacreCoeurBasilica': (48.8867, 2.3431),
            'champsElysees': (48.8698, 2.3079),
            'placeDeLaConcorde': (48.8656, 2.3211),
            'saintGermainDesPres': (48.8540, 2.3329),
            'parisOrlyAirport': (48.7262, 2.3652),
            'charlesDeGaulleAirport': (49.0097, 2.5479),
            # Add more places here if needed
        }
    elif city == 'strasbourg':
        return {
            'strasbourgCathedral': (48.5818, 7.7506),
            'europeanParliament': (48.5975, 7.7658),
            'parcOrangerie': (48.5861, 7.7760),
            'placeKleber': (48.5833, 7.7479),
            'strasbourgStation': (48.5850, 7.7348),
            'palaisRohan': (48.5793, 7.7519),
            'strasbourgAirport': (48.5383, 7.6280),
            'laPetiteFrance': (48.5800, 7.7408),
            # Add more places here if needed
        }
    elif city == 'lyon':
        return {
            'lyonCathedral': (45.7602, 4.8267),
            'placeBellecour': (45.7578, 4.8325),
            'parcTeteDor': (45.7772, 4.8558),
            'lyonPartDieuStation': (45.7606, 4.8596),
            'lyonSaintExuperyAirport': (45.7256, 5.0811),
            'museumOfFineArtsOfLyon': (45.7679, 4.8336),
            'halleTonyGarnier': (45.7331, 4.8187),
        }
    elif city == 'nantes':
        return {
            'nantesCathedral': (47.2181, -1.5534),
            'chateauDesDucsDeBretagne': (47.2155, -1.5477),
            'jardinDesPlantes': (47.2184, -1.5413),
            'nantesStation': (47.2175, -1.5419),
            'isleOfNantes': (47.2074, -1.5556),
            'nantesAirport': (47.1532, -1.6108),
            'machinesOfTheIsleOfNantes': (47.2078, -1.5645),
        }
    elif city == 'bordeaux':
        return {
            'placeDeLaBourse': (44.8412, -0.5704),
            'grandTheatre': (44.8420, -0.5746),
            'parcBordelais': (44.8483, -0.5944),
            'bordeauxSaintJeanStation': (44.8253, -0.5566),
            'bordeauxAirport': (44.8286, -0.7153),
            'pontDePierre': (44.8378, -0.5612),
            'citéDuVin': (44.8627, -0.5504),
        }
    else:
        print('invalid city')
        return None


def get_extra_attributes(input_attributes, city):
    # Coordinates for important places in Paris
    important_places = get_important_places(city)

    if important_places is not None:
        # Calculate the distances and update the input_attributes
        for place, (lat, lon) in important_places.items():
            distance = haversine(lon, lat, input_attributes['location.lon'], input_attributes['location.lat'])
            input_attributes[f'distance.{place}'] = distance
    else:
        print('important place not found')

    return input_attributes


def add_distance_features(df: pd.DataFrame, city: str):
    # Coordinates for important places in Paris
    important_places = get_important_places(city)

    if important_places is not None:
        # Iterate over each important place
        for place, (lat, lon) in important_places.items():
            # Calculate the distance from the place to each property and create a new column for it
            df[f'distance.{place}'] = (df
                                       .apply(lambda row: haversine(lon, lat, row['location.lon'], row['location.lat']),
                                              axis=1))
    else:
        print('no important places found')

    return df


def convert_bool_to_int(df: pandas.DataFrame) -> pandas.DataFrame:
    df["furnished"] = df["furnished"].astype(int)

    df["expired"] = df["expired"].astype(int)

    df["elevator"] = df["elevator"].astype(int)

    return df


def print_rows_with_nulls(df: pandas.DataFrame) -> None:
    """
    Print rows that contain null or NaN values.

    Parameters:
    - df: pandas DataFrame containing the data.
    """
    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        if row.isnull().any():
            print(f"Row {index} with NaN values:")
            print(row)
            print("\n")  # Add a newline for better readability between row


def handle_missing_values(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Handle missing values in a DataFrame:
    - Replace NaNs in numeric columns with the mean of the column.
    - Replace NaNs in object/string columns with an empty string.
    - Replace NaNs in boolean columns with False.

    Parameters:
    - df: pandas DataFrame containing the data.

    Returns:
    - pandas DataFrame: DataFrame with NaN values handled.
    """
    for column in df.columns:
        # Check if the column is numeric (int or float)
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].mean())
        # Check if the column is of object type (e.g., strings)
        elif pd.api.types.is_object_dtype(df[column]):
            df[column].fillna('', inplace=True)
        # Check if the column is boolean
        elif pd.api.types.is_bool_dtype(df[column]):
            df[column].fillna(False, inplace=True)

    return df


def filter_columns(df: pandas.DataFrame, required_columns) -> pandas.DataFrame:
    """
    Filter the DataFrame to only include the specified columns.

    Parameters:
    - df: pandas DataFrame containing the data.
    - required_columns: List of strings representing the names of the required columns.

    Returns:
    - pandas DataFrame: New DataFrame containing only the columns specified in required_columns.
    """
    # Filter the DataFrame to only include the specified columns
    filtered_df = df[required_columns]

    return filtered_df


def clean_location_data(df, lat_min, lat_max, lon_min, lon_max):
    """
    Remove rows from the DataFrame where the latitude and longitude are not within the bounds of the given city.

    Parameters:
    - df: pandas DataFrame containing the data.
    - lat_min: Minimum latitude value
    - lat_max: Maximum latitude value
    - lon_min: Minimum longitude value
    - lon_max: Maximum longitude value

    Returns:
    - Cleaned pandas DataFrame with only rows where latitude and longitude fall within the boundaries.
    """
    # Filter the DataFrame for rows where latitude and longitude fall within the specified bounds
    cleaned_df = df[
        (df['location.lat'] >= lat_min) & (df['location.lat'] <= lat_max) &
        (df['location.lon'] >= lon_min) & (df['location.lon'] <= lon_max)
        ]

    return cleaned_df
