"""
This is the main file to run the AI model for predicting the price of an apartment in a city.
"""

import json
import os
import pickle
import sys

from load_data import load_json_data, load_csv_data
from analyse_data import analyse, analyse_v2

from models.voting_regressor import voting_regressor_train_model

from models.predictor import general_predict_price
from models.xgb import xgb_train_model, xgb_print_model_with_kfold
from preprocess.preprocess import preprocess_data, preprocess_data_v2

from models.gbr import gbr_train_model, gbr_tune_hyper_parameters
from models.gbr import gbr_print_model_with_kfold

from models.nn import nn_train_model
from models.nn import nn_print_model_with_kfold

from models.random_forest import rf_train_model
from models.random_forest import rf_print_model_with_kfold

from models.knn import knn_train_model
from models.knn import knn_print_model_with_kfold

from models.lasso import lasso_train_model
from models.lasso import lasso_print_model_with_kfold
from models.lasso import lasso_tune_hyper_parameters

from preprocess.preprocess import get_extra_attributes

from data import SUPPORTED_CITIES, ENABLE_AI_DATA_SAVE
from data import VERSION

# dataset to use
# city = 'paris'
# analyse_mode = True

# TODO: Add more models here
supported_models = [
    'gbr',
    'random_forest',
    'nn',
    'knn',
    'xgb',
    'lasso',
    'voting_regressor'
]


def main():
    """
    Main function to run the AI model for predicting the price of an apartment in a city.
    :return:    None
    """
    # To get the city name from command line, use the following code
    # if not, comment it out and use the city variable above
    if len(sys.argv) < 2:
        print(f'Usage: '
              f'python3 {sys.argv[0]} <city_name> [-a] [-t <model_name>] [-p <path_to_json_file>] [-c <n_splits>] [-o]')
        print(f'Example: python3 {sys.argv[0]} paris -a -t gbr -p prediction/prediction_data.json')
        sys.exit(1)

    city = sys.argv[1]

    # Check if argument -a is given in any part of the command line
    if '-a' in sys.argv:
        analyse_mode = True
    else:
        analyse_mode = False

    # If there is an argument -t in the command line, it must contain the name of the training model after it
    # And the model name must be one of the supported models
    if '-t' in sys.argv:
        try:
            model_name = sys.argv[sys.argv.index('-t') + 1]
            if not model_name in supported_models:
                print(f"Model not supported, please use one of the following: {', '.join(supported_models)}")
                sys.exit(1)
        except IndexError:
            print(f"Please specify a model name after -t")
            sys.exit(1)
    else:
        model_name = 'gbr'
        print(f"Using default model: {model_name}")

    # check if the given city is supported
    if not city in SUPPORTED_CITIES:
        print(f"City not supported, please use one of the following: {', '.join(SUPPORTED_CITIES)}")
        sys.exit(1)

    # If there is an argument -p in the command line it must contain the path to a json file after it (prediction data)
    if '-p' in sys.argv:
        try:
            prediction_data_path = sys.argv[sys.argv.index('-p') + 1]
            try:
                # load data from the given path from json file using numpy
                with open(prediction_data_path, 'r', encoding="utf8") as file:
                    # Parse the JSON data and convert it into a Python object
                    prediction_data = json.load(file)
            except Exception as e:
                print(f"An error occurred: {e}")
                sys.exit(1)
        except IndexError:
            print(f"Please specify a path to a json file after -p")
            sys.exit(1)

    # If there is an argument -c in the command line it must contain a number after it (greater than 0), this is
    # the number of KFold Cross Validation splits (the command -c is optional)
    n_splits = -1
    if '-c' in sys.argv:
        try:
            n_splits = int(sys.argv[sys.argv.index('-c') + 1])
            if n_splits <= 0:
                print(f"Please specify a number greater than 0 after -c")
                sys.exit(1)
        except IndexError:
            print(f"Please specify a number after -c")
            sys.exit(1)
        except ValueError:
            print(f"Please specify a number after -c")
            sys.exit(1)

    # Path to the data file
    file_path = f"data/data-{city}.json"

    # Check if file exists
    try:
        f = open(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit(1)

    print(f"Using data file: {file_path}")
    print('#############################################' + '\n')

    # Load the data as pandas DataFrame
    df = load_json_data(file_path)

    print(f"Loaded data for {city}")
    print('#############################################' + '\n')

    cleaned_df = preprocess_data(df, city)

    print(f"Preprocessed data for {city}")
    print('#############################################' + '\n')

    # Show the analysis of the data
    if analyse_mode:
        analyse(cleaned_df)
        print('#############################################' + '\n')

    # If there is an argument -o in the command line it must not contain any other arguments
    # This is the argument for hyperparameter tuning and optimization (the command -o is optional)
    # TODO: Add more models here
    if '-o' in sys.argv:
        print(f"Hyperparameter tuning and optimization for model: {model_name}...")
        if model_name == 'gbr':
            gbr_tune_hyper_parameters(cleaned_df)
            print('#############################################' + '\n')
        if model_name == 'lasso':
            lasso_tune_hyper_parameters(cleaned_df)
            print('#############################################' + '\n')
        else:
            print(f"Model not supported yet for hyperparameter tuning")
            print('#############################################' + '\n')

    # Check if the model is already trained
    filename = model_name + '.sav'

    # If file does not exist, train the model
    if not os.path.isfile(f'tmp_ai/{filename}'):
        print(f"Training model: {model_name}...")
        # cleaned_df is now ready to be trained
        # TODO: Add more models here
        if model_name == 'gbr':
            model = gbr_train_model(cleaned_df)
        elif model_name == 'random_forest':
            model = rf_train_model(cleaned_df)
        elif model_name == 'nn':
            model = nn_train_model(cleaned_df)
        elif model_name == 'knn':
            model = knn_train_model(cleaned_df)
        elif model_name == 'lasso':
            model = lasso_train_model(cleaned_df)
        elif model_name == 'xgb':
            model = xgb_train_model(cleaned_df)
        elif model_name == 'voting_regressor':
            model = voting_regressor_train_model(cleaned_df)
        else:
            print(f"Model not supported, please use one of the following: {', '.join(supported_models)}")
            sys.exit(1)

        if ENABLE_AI_DATA_SAVE:
            save_model_to_file(filename, model)

        print(f"Trained model: {model_name}")
        print('#############################################' + '\n')
    else:
        print(f"Model already trained: {model_name}. Loading from file...")
        print('#############################################' + '\n')
        model = load_model_from_file(filename)

    # Use KFold Cross Validation to calculate mean, precision, etc.
    # TODO: Add more models here
    if n_splits > 0:
        print(f"Calculating mean, precision, etc. using KFold Cross Validation with {n_splits} splits")
        if model_name == 'gbr':
            gbr_print_model_with_kfold(cleaned_df, n_splits)
        elif model_name == 'random_forest':
            rf_print_model_with_kfold(cleaned_df, n_splits)
        elif model_name == 'nn':
            nn_print_model_with_kfold(cleaned_df, n_splits)
        elif model_name == 'knn':
            knn_print_model_with_kfold(cleaned_df, n_splits)
        elif model_name == 'xgb':
            xgb_print_model_with_kfold(cleaned_df, n_splits)
        elif model_name == 'lasso':
            lasso_print_model_with_kfold(cleaned_df, n_splits)
        else:
            print(f"Model not supported, please use one of the following: {', '.join(supported_models)}")
            sys.exit(1)
        print('#############################################' + '\n')

    # Predict the price of the apartment using the trained model and the given input attributes
    # check if the prediction data is given
    if '-p' in sys.argv:
        i = 1
        # parse the prediction data, for each element in the array, get the attributes and predict the price
        for prediction in prediction_data['elements']:
            input_attributes = prediction
            # Get additional attributes (distances to important places)
            input_attributes = get_extra_attributes(input_attributes, city)
            # The prediction is general to all models
            predicted_price = general_predict_price(model, input_attributes)
            print(f"{i}. The predicted price of the apartment is: {predicted_price}")
            i += 1
        print('#############################################' + '\n')


def main_v2():
    """
    Main function to run the AI model for predicting the price of an apartment in a city (Version 2).
    :return:    None
    """
    # In Version 2, The city name is not given
    if len(sys.argv) < 2:
        print(f'Usage: '
              f'python3 {sys.argv[0]} <city_name> [-a] [-t <model_name>] [-p <path_to_json_file>] [-c <n_splits>] [-o]')
        print(f'Example: python3 {sys.argv[0]} paris -a -t gbr -p prediction/prediction_data.json')
        sys.exit(1)

    city = sys.argv[1]

    # check if the given city is supported
    if not city in SUPPORTED_CITIES:
        print(f"City not supported, please use one of the following: {', '.join(SUPPORTED_CITIES)}")
        sys.exit(1)

    # Check if argument -a is given in any part of the command line
    if '-a' in sys.argv:
        analyse_mode = True
    else:
        analyse_mode = False

    # If there is an argument -t in the command line, it must contain the name of the training model after it
    # And the model name must be one of the supported models
    if '-t' in sys.argv:
        try:
            model_name = sys.argv[sys.argv.index('-t') + 1]
            if not model_name in supported_models:
                print(f"Model not supported, please use one of the following: {', '.join(supported_models)}")
                sys.exit(1)
        except IndexError:
            print(f"Please specify a model name after -t")
            sys.exit(1)
    else:
        model_name = 'gbr'
        print(f"Using default model: {model_name}")

    # If there is an argument -p in the command line it must contain the path to a json file after it (prediction data)
    if '-p' in sys.argv:
        try:
            prediction_data_path = sys.argv[sys.argv.index('-p') + 1]
            try:
                # load data from the given path from json file using numpy
                with open(prediction_data_path, 'r', encoding="utf8") as file:
                    # Parse the JSON data and convert it into a Python object
                    prediction_data = json.load(file)
            except Exception as e:
                print(f"An error occurred: {e}")
                sys.exit(1)
        except IndexError:
            print(f"Please specify a path to a json file after -p")
            sys.exit(1)

    # If there is an argument -c in the command line it must contain a number after it (greater than 0), this is
    # the number of KFold Cross Validation splits (the command -c is optional)
    n_splits = -1
    if '-c' in sys.argv:
        try:
            n_splits = int(sys.argv[sys.argv.index('-c') + 1])
            if n_splits <= 0:
                print(f"Please specify a number greater than 0 after -c")
                sys.exit(1)
        except IndexError:
            print(f"Please specify a number after -c")
            sys.exit(1)
        except ValueError:
            print(f"Please specify a number after -c")
            sys.exit(1)

    # get the valuers foncieres data path
    # TODO: Check the file name
    file_path = f"data/valeursfoncieres-2023_{city}.csv"

    # Check if file exists
    try:
        f = open(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit(1)

    print(f"Using data file: {file_path}")
    print('#############################################' + '\n')

    # TODO: Attention to the separator, check the data file
    df = load_csv_data(file_path, separator=',')

    print(f"Loaded data {file_path}")
    print('#############################################' + '\n')

    cleaned_df = preprocess_data_v2(df, city)

    print(f"Preprocessed data {file_path}")
    print('#############################################' + '\n')

    # Show the analysis of the data
    if analyse_mode:
        analyse_v2(cleaned_df)
        print('#############################################' + '\n')

    # If there is an argument -o in the command line it must not contain any other arguments
    # This is the argument for hyperparameter tuning and optimization (the command -o is optional)
    # TODO: Add more models here
    if '-o' in sys.argv:
        print(f"Hyperparameter tuning and optimization for model: {model_name}...")
        if model_name == 'gbr':
            gbr_tune_hyper_parameters(cleaned_df)
            print('#############################################' + '\n')
        if model_name == 'lasso':
            lasso_tune_hyper_parameters(cleaned_df)
            print('#############################################' + '\n')
        else:
            print(f"Model not supported yet for hyperparameter tuning")
            print('#############################################' + '\n')

    # Check if the model is already trained
    filename = model_name + '.sav'

    # If file does not exist, train the model
    if not os.path.isfile(f'tmp_ai/{filename}'):
        print(f"Training model: {model_name}...")

        # cleaned_df is now ready to be trained
        # TODO: Add more models here
        if model_name == 'gbr':
            model = gbr_train_model(cleaned_df)
        elif model_name == 'random_forest':
            model = rf_train_model(cleaned_df)
        elif model_name == 'nn':
            model = nn_train_model(cleaned_df)
        elif model_name == 'knn':
            model = knn_train_model(cleaned_df)
        elif model_name == 'lasso':
            model = lasso_train_model(cleaned_df)
        elif model_name == 'xgb':
            model = xgb_train_model(cleaned_df)
        else:
            print(f"Model not supported, please use one of the following: {', '.join(supported_models)}")
            sys.exit(1)

        print(f"Trained model: {model_name}")
        print('#############################################' + '\n')

        if ENABLE_AI_DATA_SAVE:
            save_model_to_file(filename, model)

    else:
        print(f"Model already trained: {model_name}. Loading from file...")
        print('#############################################' + '\n')
        model = load_model_from_file(filename)

    # Use KFold Cross Validation to calculate mean, precision, etc.
    # TODO: Add more models here
    if n_splits > 0:
        print(f"Calculating mean, precision, etc. using KFold Cross Validation with {n_splits} splits")
        if model_name == 'gbr':
            gbr_print_model_with_kfold(cleaned_df, n_splits)
        elif model_name == 'random_forest':
            rf_print_model_with_kfold(cleaned_df, n_splits)
        elif model_name == 'nn':
            nn_print_model_with_kfold(cleaned_df, n_splits)
        elif model_name == 'knn':
            knn_print_model_with_kfold(cleaned_df, n_splits)
        elif model_name == 'xgb':
            xgb_print_model_with_kfold(cleaned_df, n_splits)
        elif model_name == 'lasso':
            lasso_print_model_with_kfold(cleaned_df, n_splits)
        else:
            print(f"Model not supported, please use one of the following: {', '.join(supported_models)}")
            sys.exit(1)
        print('#############################################' + '\n')


def save_model_to_file(filename, clf):
    """
    Save the trained model to a file.
    :param filename:    Name of the file to save the model to
    :param clf:         Trained model
    :return:            None
    """
    # create tmp_ai folder if it doesn't exist in the current directory
    if not os.path.exists('tmp_ai'):
        os.makedirs('tmp_ai')

    with open(f'tmp_ai/{filename}', 'wb') as fout:
        pickle.dump(clf, fout)
        print("Model saved to " + f'tmp_ai/{filename}')


def load_model_from_file(filename):
    """
    Load the trained model from a file.
    :param filename:    Name of the file to load the model from
    :return:            Trained model
    """
    # create tmp_ai folder if it doesn't exist in the current directory
    if not os.path.exists('tmp_ai'):
        os.makedirs('tmp_ai')

    print("Loading model from tmp_ai/" + filename)
    with open(f'tmp_ai/{filename}', 'rb') as f:
        clf = pickle.load(f)
        return clf


if __name__ == "__main__":
    if VERSION == 1:
        main()
    if VERSION == 2:
        main_v2()
