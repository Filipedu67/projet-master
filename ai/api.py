from flask import Flask, request, jsonify

from ai.load_data import load_json_data
from models.gbr import gbr_train_model
from models.predictor import general_predict_price
from preprocess.preprocess import preprocess_data, get_extra_attributes

supported_cities = [
    'bordeaux',
    'lille',
    'lyon',
    'marseille',
    'montpellier',
    'nantes',
    'nice',
    'paris',
    'strasbourg',
    'toulouse'
]

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Extract JSON data from the request
    data = request.json

    # You can now use this 'data' object to get the input for your model
    city = data.get('city', 'paris')
    # TODO: Choose the best model here
    # model_name = data.get('model', 'gbr')

    # Ensure city and model are supported
    if city not in supported_cities:
        return jsonify({'error': 'Unsupported city'}), 400

    # Path to the data file
    file_path = f"data/data-{city}.json"

    # Check if file exists
    try:
        f = open(file_path)
    except FileNotFoundError:
        return jsonify({'error': f"File not found: {file_path}"}), 500

    # Load the data as pandas DataFrame
    df = load_json_data(file_path)

    # Preprocess the data
    cleaned_df = preprocess_data(df, city)

    # cleaned_df is now ready to be trained
    # TODO: Replace the model with the best possible model
    model = gbr_train_model(cleaned_df)

    # Do the prediction
    # TODO: Do it a few times and get a mean ?
    # input_attributes is all the data except the city
    input_attributes = data.copy()
    del input_attributes['city']
    # Get additional attributes (distances to important places)
    input_attributes = get_extra_attributes(input_attributes, city)
    # The prediction is general to all models
    predicted_price = general_predict_price(model, input_attributes)

    return jsonify({'predicted_price': predicted_price}), 200


if __name__ == "__main__":
    app.run(debug=True)
