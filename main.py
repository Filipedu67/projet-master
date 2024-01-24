import sys

from load_data import load_data
from analyse_data import analyse
from preprocess.preprocess import preprocess_data

city = 'paris'


def main():
    # To get the city name from command line, uncomment the following code
    # if len(sys.argv) < 2:
    #     print(f'Usage: python3 {sys.argv[0]} <city_name>')
    #     sys.exit(1)
    #
    # city = sys.argv[1]

    file_path = f"data/data-{city}.json"

    # Load the data
    df = load_data(file_path)

    cleaned_df = preprocess_data(df, city)

    # Show the analysis of the data
    analyse(cleaned_df)

    # df_preprocessed is now ready for further analysis or modeling


if __name__ == "__main__":
    main()
