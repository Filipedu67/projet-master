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
        print("Invalid city")
        return None

    return clean_location_data(df, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)


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
        (df['location.lat'] >= lat_min) & (df['location.lat'] <= lat_max) &
        (df['location.lon'] >= lon_min) & (df['location.lon'] <= lon_max)
        ]

    return cleaned_df
