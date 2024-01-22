from load_data import load_data
from flatten_data import flatten_data
from preprocess_data import preprocess_data

file_path = 'data/cities/data-paris.json' # Update this to the path of your JSON file

# Load the data
df = load_data(file_path)



# Flatten the data if necessary
# df_flattened = flatten_data(df)

# Preprocess the data
df_preprocessed = preprocess_data(df)

# df_preprocessed is now ready for further analysis or modeling
