from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer


def preprocess_textual_data(data):
    # Implement textual data preprocessing
    return data


def preprocess_listlike_data(data):
    # Implement list-like data preprocessing
    return data


def preprocess_data(df):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), ['bedroom', 'floor', 'price', 'pricePerMeter', 'room', 'surface']),
            ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent', missing_values=None)),
                                    ('to_string', FunctionTransformer(lambda x: x.astype(str)))]),
             ['elevator', 'expired', 'furnished', 'transactionType']),
            ('text', FunctionTransformer(preprocess_textual_data), ['description', 'title']),
            ('listlike', FunctionTransformer(preprocess_listlike_data), ['pictures', 'publisherTypes', 'stations'])
        ]
    )

    return preprocessor.fit_transform(df)
