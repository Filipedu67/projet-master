import pandas
import pandas as pd
from math import radians, cos, sin, asin, sqrt

from sklearn.preprocessing import LabelEncoder

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

# Define the bounds for Lille (approximate values)
LILLE_LAT_MIN = 50.594340
LILLE_LAT_MAX = 50.677521
LILLE_LON_MIN = 2.952042
LILLE_LON_MAX = 3.153229

# # Define the bounds for Marseille (approximate values)
MARS_LAT_MIN = 43.155416
MARS_LAT_MAX = 43.414335
MARS_LON_MIN = 5.221784
MARS_LON_MAX = 5.559875

# # Define the bounds for Montpellier (approximate values)
MONT_LAT_MIN = 43.540333
MONT_LAT_MAX = 43.673958
MONT_LON_MIN = 3.769510
MONT_LON_MAX = 3.984958

# # Define the bounds for Nice (approximate values)
NICE_LAT_MIN = 43.630287
NICE_LAT_MAX = 43.722495
NICE_LON_MIN = 7.148702
NICE_LON_MAX = 7.332597

# Define the bounds for Toulouse (approximate values)
TOUL_LAT_MIN = 43.490621
TOUL_LAT_MAX = 43.698439
TOUL_LON_MIN = 1.324922
TOUL_LON_MAX = 1.536952

COLUMN_TO_PREDICT = 'Valeur fonciere'

# Define the columns you want to keep
# IMPORTANT: When you add new columns, remember to handle their value type (conversion to int, etc.)
COLUMNS_TO_KEEP = ['price', 'elevator', 'location.lat', 'location.lon', 'surface', 'bedroom', 'floor',
                   'furnished', 'room', 'propertyType', 'city.department.code']

COLUMNS_TO_KEEP_V2 = ['Valeur fonciere', 'No voie', 'B/T/Q', 'Type de voie',
                      'Code voie', 'Voie',
                      'Code postal', 'Code departement', 'Code commune', 'Commune',
                      '1er lot',
                      'Surface Carrez du 1er lot', '2eme lot', 'Surface Carrez du 2eme lot', '3eme lot',
                      'Surface Carrez du 3eme lot',
                      '4eme lot', 'Surface Carrez du 4eme lot', '5eme lot', 'Surface Carrez du 5eme lot',
                      'Nombre de lots',
                      'Type local', 'Code type local', 'Surface reelle bati',
                      'Nombre pieces principales',
                      'Surface terrain', 'No disposition', 'Nature mutation', 'Prefixe de section', 'Section', 'No plan',
                      'Nature culture', 'Nature culture speciale']
# Identifiant local

# minimum and maximum price threshold
PRICE_THRESHOLD = [100000, 300000]

ADD_METRO_STATION = False


def preprocess_data(df: pandas.DataFrame, city: str) -> pandas.DataFrame:
    """
    Preprocess the data by cleaning it up and adding new features.
    :param df: pandas DataFrame containing the data.
    :param city: Name of the city.
    :return: pandas DataFrame with cleaned up data and new features added.
    """
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
    elif city == "lille":
        lat_min = LILLE_LAT_MIN
        lon_min = LILLE_LON_MIN
        lat_max = LILLE_LAT_MAX
        lon_max = LILLE_LON_MAX
    elif city == "marseille":
        lat_min = MARS_LAT_MIN
        lon_min = MARS_LON_MIN
        lat_max = MARS_LAT_MAX
        lon_max = MARS_LON_MAX
    elif city == "montpellier":
        lat_min = MONT_LAT_MIN
        lon_min = MONT_LON_MIN
        lat_max = MONT_LAT_MAX
        lon_max = MONT_LON_MAX
    elif city == "nice":
        lat_min = NICE_LAT_MIN
        lon_min = NICE_LON_MIN
        lat_max = NICE_LAT_MAX
        lon_max = NICE_LON_MAX
    elif city == "toulouse":
        lat_min = TOUL_LAT_MIN
        lon_min = TOUL_LON_MIN
        lat_max = TOUL_LAT_MAX
        lon_max = TOUL_LON_MAX
    else:
        print("Invalid city")
        return None

    # Starting data clean up

    df = convert_bool_to_int(df)

    df = filter_columns(df, COLUMNS_TO_KEEP)

    # There are some rows with repeated longitude values, which are most likely errors
    # We will remove these rows
    df = clean_data_with_repeated_longitudes(df)

    # Remove rows where the latitude and longitude are not within the bounds of the given city
    df = clean_location_data(df, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)

    # Handle missing values by replacing them with the mean, empty string, or False
    df = handle_missing_values(df)

    df = limit_price(df, PRICE_THRESHOLD)

    # Add extra attributes (e.g., distance to important places)
    df = add_distance_features(df, city)

    # Print rows with null or NaN values
    print_rows_with_nulls(df)

    # end of cleaning up the data

    return df


def limit_price(df: pandas.DataFrame, threshold: list) -> pandas.DataFrame:
    """
    Limit the price of the properties to the given threshold.

    :param df: pandas DataFrame containing the data.
    :param threshold: List containing the minimum and maximum price threshold.
    :return: pandas DataFrame with price limited to the given threshold.
    """

    print(f"Limiting the price of the properties to the range {threshold[0]} - {threshold[1]}")
    print("#####################################################")

    # Limit the price of the properties to the given threshold
    df = df[(df[COLUMN_TO_PREDICT] >= threshold[0]) & (df[COLUMN_TO_PREDICT] <= threshold[1])]

    return df


def preprocess_data_v2(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Preprocess the data by cleaning it up and adding new features.
    :param df: pandas DataFrame containing the data.
    :param city: Name of the city.
    :return: pandas DataFrame with cleaned up data and new features added.
    """
    # Starting data clean up

    df = filter_columns(df, COLUMNS_TO_KEEP_V2)

    # Handle missing values by replacing them with the mean, empty string, or False
    df = handle_missing_values_v2(df)

    # Limit the price of the properties to the given threshold
    df = limit_price(df, PRICE_THRESHOLD)

    # Convert strings to integers
    df = label_encode_data(df)

    # Print rows with null or NaN values
    print_rows_with_nulls(df)

    # end of cleaning up the data
    return df


def label_encode_data(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Convert string values to integers.
    :param df: pandas DataFrame containing the data.
    :return: pandas DataFrame with string values converted to integers.
    """

    # Initialize label encoders
    le_nature_mutation = LabelEncoder()
    le_type_de_voie = LabelEncoder()
    le_commune = LabelEncoder()
    le_type_local = LabelEncoder()
    btq = LabelEncoder()
    code_voie = LabelEncoder()
    voie = LabelEncoder()
    section = LabelEncoder()
    nature_culture = LabelEncoder()
    nature_culture_speciale = LabelEncoder()

    # Fit and transform the data using .loc for explicit in-place modification
    if 'Nature mutation' in df.columns:
        df.loc[:, 'Nature mutation'] = le_nature_mutation.fit_transform(df['Nature mutation'])

    if 'B/T/Q' in df.columns:
        df.loc[:, 'B/T/Q'] = btq.fit_transform(df['B/T/Q'])

    if 'Type de voie' in df.columns:
        df.loc[:, 'Type de voie'] = le_type_de_voie.fit_transform(df['Type de voie'])

    if 'Code voie' in df.columns:
        df.loc[:, 'Code voie'] = code_voie.fit_transform(df['Code voie'])

    if 'Voie' in df.columns:
        df.loc[:, 'Voie'] = voie.fit_transform(df['Voie'])

    if 'Commune' in df.columns:
        df.loc[:, 'Commune'] = le_commune.fit_transform(df['Commune'])

    if 'Section' in df.columns:
        df.loc[:, 'Section'] = section.fit_transform(df['Section'])

    if 'Type local' in df.columns:
        df.loc[:, 'Type local'] = le_type_local.fit_transform(df['Type local'])

    if 'Nature culture' in df.columns:
        df.loc[:, 'Nature culture'] = nature_culture.fit_transform(df['Nature culture'])

    if 'Nature culture speciale' in df.columns:
        df.loc[:, 'Nature culture speciale'] = nature_culture_speciale.fit_transform(df['Nature culture speciale'])

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


def get_metro_stations_for_city(city_name):
    """
    Extracts metro station coordinates for a specified city from the given dataset.

    Parameters:
    - city_name (str): The name of the city for which to extract metro station coordinates.
    - metro_data (pd.DataFrame): DataFrame containing metro station data.

    Returns:
    - List of tuples: Each tuple contains (longitude, latitude) of a metro station in the specified city.
    """

    # Load the metro station data
    metro_data_path = 'data/metro-france.csv'
    metro_df = pd.read_csv(metro_data_path, sep=';')

    # Filter the dataframe for the specified city
    # city_metro_stations = metro_df[metro_df['Commune nom'].str.lower() == city_name.lower()]
    city_metro_stations = metro_df[metro_df['Commune nom'].str.contains(city_name, case=False, na=False)]

    # Extract the coordinates as a list of tuples
    station_coordinates = list(city_metro_stations[['Longitude', 'Latitude']].itertuples(index=False, name=None))

    return station_coordinates


def get_important_places(city):
    """
    Return a dictionary of important places in the given city.
    :param city: city name
    :return: dictionary of important places
    """

    metro_station_places = get_metro_stations_for_city(city)
    # Convert metro stations list to a dictionary with unique keys
    metro_stations_dict = {f"metroStation{i + 1}": station for i, station in enumerate(metro_station_places)}

    if city == 'paris':
        important_places = {
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
        important_places = {
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
        important_places = {
            'lyonCathedral': (45.7602, 4.8267),
            'placeBellecour': (45.7578, 4.8325),
            'parcTeteDor': (45.7772, 4.8558),
            'lyonPartDieuStation': (45.7606, 4.8596),
            'lyonSaintExuperyAirport': (45.7256, 5.0811),
            'museumOfFineArtsOfLyon': (45.7679, 4.8336),
            'halleTonyGarnier': (45.7331, 4.8187),
        }
    elif city == 'nantes':
        important_places = {
            'nantesCathedral': (47.2181, -1.5534),
            'chateauDesDucsDeBretagne': (47.2155, -1.5477),
            'jardinDesPlantes': (47.2184, -1.5413),
            'nantesStation': (47.2175, -1.5419),
            'isleOfNantes': (47.2074, -1.5556),
            'nantesAirport': (47.1532, -1.6108),
            'machinesOfTheIsleOfNantes': (47.2078, -1.5645),
        }
    elif city == 'bordeaux':
        important_places = {
            'placeDeLaBourse': (44.8412, -0.5704),
            'grandTheatre': (44.8420, -0.5746),
            'parcBordelais': (44.8483, -0.5944),
            'bordeauxSaintJeanStation': (44.8253, -0.5566),
            'bordeauxAirport': (44.8286, -0.7153),
            'pontDePierre': (44.8378, -0.5612),
            'citéDuVin': (44.8627, -0.5504),
        }
    elif city == 'lille':
        important_places = {
            'lilleFlandresStation': (50.6366, 3.0695),
            'grandPlaceLille': (50.6372, 3.0633),
            'lilleEuropeStation': (50.6394, 3.0755),
            'palaisDesBeauxArts': (50.6322, 3.0603),
            'lilleZoo': (50.6325, 3.0422),
            'lilleAirport': (50.5617, 3.0894),
        }
    elif city == 'marseille':
        important_places = {
            'oldPortOfMarseille': (43.2954, 5.3745),
            'notreDameDeLaGarde': (43.2849, 5.3698),
            'stadeVelodrome': (43.2706, 5.3959),
            'marseilleSaintCharlesStation': (43.3027, 5.3806),
            'calanquesNationalPark': (43.2136, 5.4534),
            'marseilleProvenceAirport': (43.4393, 5.2214),
        }
    elif city == 'montpellier':
        important_places = {
            'placeDeLaComedie': (43.6085, 3.8795),
            'montpellierSaintRochStation': (43.6045, 3.8802),
            'montpellierMediterraneeAirport': (43.5762, 3.9631),
            'fabreMuseum': (43.6114, 3.8773),
            'aqueductSaintClement': (43.6156, 3.8694),
            'montpellierZoo': (43.6391, 3.8739),
        }
    elif city == 'nice':
        important_places = {
            'promenadeDesAnglais': (43.6952, 7.2656),
            'niceVilleStation': (43.7045, 7.2619),
            'castleHillOfNice': (43.6951, 7.2798),
            'coursSaleyaMarket': (43.6950, 7.2754),
            'niceCoteDAzurAirport': (43.6653, 7.2150),
            'matisseMuseum': (43.7192, 7.2763),
        }
    elif city == 'toulouse':
        important_places = {
            'capitoleDeToulouse': (43.6045, 1.4435),
            'toulouseMatabiauStation': (43.6111, 1.4544),
            'citéDeLEspace': (43.5861, 1.4904),
            'basilicaOfSaintSernin': (43.6082, 1.4429),
            'toulouseBlagnacAirport': (43.6293, 1.3636),
            'garonneRiverfront': (43.5985, 1.4430),
        }
    else:
        print('invalid city')
        return None

    # Combine the metro stations with the important places
    # NOTE: Doesn't work well with NN model
    if ADD_METRO_STATION:
        important_places.update(metro_stations_dict)

    return important_places


def get_extra_attributes(input_attributes, city):
    """
    Add extra attributes to the input_attributes dictionary.
    :param input_attributes: existing attributes
    :param city: city name
    :return: input_attributes with extra attributes
    """

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
    """
    Add distance features to the DataFrame.
    :param df: pandas DataFrame containing the data.
    :param city: Name of the city.
    :return: pandas DataFrame with distance features added.
    """
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
    """
    Convert boolean values to integers (0 or 1).
    :param df: pandas DataFrame containing the data.
    :return: pandas DataFrame with boolean values converted to integers.
    """
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


def handle_missing_values_v2(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Handle missing values in a DataFrame:
    - Delete rows where 'Valeur fonciere' is NaN.
    - Replace NaNs in numeric columns with the mean of the column.
    - Replace NaNs in object/string columns with an empty string.
    - Replace NaNs in boolean columns with False.

    Parameters:
    - df: pandas DataFrame containing the data.

    Returns:
    - pandas DataFrame: DataFrame with NaN values handled.
    """
    # Count rows before deletion
    initial_row_count = len(df)

    # Delete rows where 'Valeur fonciere' is NaN
    df = df.dropna(subset=['Valeur fonciere'])

    # Count rows after deletion to calculate the number of deleted rows
    final_row_count = len(df)
    deleted_rows = initial_row_count - final_row_count
    print(f'Number of deleted rows due to valuer fonciere column being empty: {deleted_rows}')
    print(f'#####################################################')

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            if df[column].isnull().all():
                # Directly assign the filled column to the DataFrame
                df.loc[:, column] = df.loc[:, column].fillna(0)  # For columns with all NaN values
            else:
                # Calculate the mean and directly assign the filled column to the DataFrame
                column_mean = df.loc[:, column].mean()
                df.loc[:, column] = df.loc[:, column].fillna(column_mean)
        elif pd.api.types.is_object_dtype(df[column]):
            # Directly assign the filled column with empty string for object-type columns
            df.loc[:, column] = df.loc[:, column].fillna('')
        elif pd.api.types.is_bool_dtype(df[column]):
            # Directly assign the filled column with False for boolean-type columns
            df.loc[:, column] = df.loc[:, column].fillna(False)
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            # For datetime columns, you can fill with a specific placeholder date or leave as is
            # Example: df.loc[:, column] = df.loc[:, column].fillna(pd.Timestamp('your_placeholder_date'))
            pass

    return df


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
            df.loc[:, column] = df[column].fillna(df[column].mean())
        # Check if the column is of object type (e.g., strings)
        elif pd.api.types.is_object_dtype(df[column]):
            df.loc[:, column] = df[column].fillna('')
        # Check if the column is boolean
        elif pd.api.types.is_bool_dtype(df[column]):
            df.loc[:, column] = df[column].fillna(False)

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
