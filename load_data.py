import json
import pandas as pd


# Replace 'your_file.json' with your actual file path
def load_data(file_path):
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

            # Convert 'elevator' column to boolean
            df['elevator'] = df['elevator'].astype(bool)

            # Convert 'price' column to numeric and explicitly remove NaN values
            df['price'] = pd.to_numeric(df['price'], errors='coerce')

            df['bedroom'] = pd.to_numeric(df['bedroom'], errors='coerce')

            df['description'] = df['description'].astype(str)

            df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce', utc=True)

            df['expired'] = df['expired'].astype(bool)

            df['floor'] = pd.to_numeric(df['floor'], errors='coerce')

            df['furnished'] = df['furnished'].astype(bool)

            df['pricePerMeter'] = pd.to_numeric(df['pricePerMeter'], errors='coerce')

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

    except json.JSONDecodeError as err:
        print(f"Error parsing JSON: {err}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
