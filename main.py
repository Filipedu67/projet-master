from load_data import load_data
from analyse_data import analyse

file_path = 'data/cities/data-paris.json' # Update this to the path of your JSON file

# Load the data
df = load_data(file_path)

# Show the analysis of the data
analyse(df)

# Preprocess the data
# df_preprocessed = preprocess_data(df)

# df_preprocessed is now ready for further analysis or modeling
