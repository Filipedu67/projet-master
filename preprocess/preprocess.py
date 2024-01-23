import pandas as pd

# Define the bounds for Paris (Ile de france) (approximate values)
PARIS_LAT_MIN = 48.851981
PARIS_LAT_MAX = 48.997016
PARIS_LON_MIN = 1.682391
PARIS_LON_MAX = 2.532633


def preprocess_data(df):
    clean_location_data(df, lat_min=PARIS_LAT_MIN, lat_max=PARIS_LAT_MAX, lon_min=PARIS_LON_MIN, lon_max=PARIS_LON_MAX)


def clean_location_data(df, lat_min, lat_max, lon_min, lon_max):
    """
    Remove rows from the DataFrame where the latitude and longitude are not within the bounds of Paris.

    Parameters:
    - df: pandas DataFrame containing the data.
    - lat_min: Minimum latitude value for Paris.
    - lat_max: Maximum latitude value for Paris.
    - lon_min: Minimum longitude value for Paris.
    - lon_max: Maximum longitude value for Paris.

    Returns:
    - Cleaned pandas DataFrame with only rows where latitude and longitude fall within Paris.
    """
    # Filter the DataFrame for rows where latitude and longitude fall within the specified bounds
    cleaned_df = df[
        (df['lat'] >= lat_min) & (df['lat'] <= lat_max) &
        (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
        ]

    return cleaned_df


# Assuming `df` is your DataFrame and it has 'lat' and 'lon' columns for latitude and longitude
# And assuming you have the correct min and max values for latitude and longitude for Paris


# Clean the DataFrame
cleaned_df = clean_location_data(df, PARIS_LAT_MIN, PARIS_LAT_MAX, PARIS_LON_MIN, PARIS_LON_MAX)

# Now `cleaned_df` is the DataFrame with only the rows that have latitude and longitude within Paris
