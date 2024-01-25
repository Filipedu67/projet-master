import sys

from load_data import load_data
from analyse_data import analyse
from preprocess.preprocess import preprocess_data
from models.gbr import train_model
from models.gbr import print_model_with_kfold
# from models.random_forest import train_model
from models.gbr import predict_price
from preprocess.preprocess import get_extra_attributes

# dataset to use
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
    # TODO: Add option to turn this on or off
    analyse(cleaned_df)

    # cleaned_df is now ready to be trained
    model = train_model(cleaned_df)

    # Uncomment this to use KFold Cross Validation to calculate mean, precision, etc.
    print_model_with_kfold(cleaned_df, 30)

    # Example usage:
    input_attributes = {
        'surface': 50,
        'location.lat': 48.8566,
        'location.lon': 2.4522,
        'bedroom': 1,
        'floor': 10,
        'furnished': 1,
        'room': 2,
        'elevator': 0
    }

    # Get additional attributes (distances to important places)
    input_attributes = get_extra_attributes(input_attributes, city)

    predicted_price = predict_price(model, input_attributes)
    print(f"The predicted price of the apartment is: {predicted_price}")


if __name__ == "__main__":
    main()
