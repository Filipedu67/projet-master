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
            return df

    except json.JSONDecodeError as err:
        print(f"Error parsing JSON: {err}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
