import sys

from load_data import load_data
from analyse_data import analyse
from preprocess.preprocess import preprocess_data

city = 'paris'
file_path = f'data/data-{city}.json'  # Update this to the path of your JSON file


def main():
    # To get the city name from command line, uncomment the following code
    # if len(sys.argv) < 2:
    #     print(f'Usage: python3 {sys.argv[0]} <city_name>')
    #     sys.exit(1)
    #
    # city = sys.argv[1]

    # file_path = f"data/data-{city}.json"

    # Load the data
    df = load_data(file_path)

    # Show the analysis of the data
    analyse(df)

    # TODO: Preprocess the data


if __name__ == "__main__":
    main()
