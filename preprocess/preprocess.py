import pandas
import pandas as pd

# Define the bounds for Paris (Ile de france) (approximate values)
PARIS_LAT_MIN = 48.851981
PARIS_LAT_MAX = 48.997016
PARIS_LON_MIN = 1.682391
PARIS_LON_MAX = 2.532633

# Define the bounds for Strasbourg (approximate values)
STRA_LAT_MIN = 48.573826
STRA_LAT_MAX = 48.600548
STRA_LON_MIN = 7.721565
STRA_LON_MAX = 7.722536

# Define the columns you want to keep
COLUMNS_TO_KEEP = ['price', 'elevator', 'location.lat', 'location.lon',
                   'surface', 'bedroom', 'createdAt', 'description', 'floor',
                   'furnished', 'pricePerMeter', 'room', 'title']  # Add other column names as needed


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
    else:
        # TODO: Add more cities
        print("Invalid city")
        return None

    # Starting data clean up

    df = filter_columns(df, COLUMNS_TO_KEEP)

    df = clean_location_data(df, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)

    df = handle_missing_values(df)

    print_rows_with_nulls(df)

    # end of cleaning up the data

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
            df[column].fillna(df[column].mean(), inplace=True)
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
