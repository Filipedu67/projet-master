import pandas as pd


def flatten_data(json_data):
    flattened_data = pd.json_normalize(
        json_data,
        record_prefix='nested_',
        record_path=['adverts'],
        meta=[
            'createdAt',  # Main record's createdAt
            ['city.location.lat', 'city_lat'],  # Renamed with prefix for clarity
            ['city.location.lon', 'city_lon'],  # Renamed with prefix for clarity
            ['city.name', 'city_name'],  # Renamed with prefix for clarity
            ['city.zipcode', 'city_zipcode'],  # Renamed with prefix for clarity
            ['city.department.code', 'department_code'],  # Renamed with prefix for clarity
            ['city.department.name', 'department_name'],  # Renamed with prefix for clarity
            ['city.region.name', 'region_name']  # Renamed with prefix for clarity
        ],
        errors='ignore'
    )
    return flattened_data
