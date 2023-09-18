import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import pytest
import src.common_functions
from joblib import load


@pytest.fixture
def data():
    """
    Load dataset
    """
    df = pd.read_csv('data/census.csv')
    return df

def test_inference_above():
    """
    Check inference performance
    """
    model = load("data/model/model.joblib")
    encoder = load("data/model/encoder.joblib")
    lb = load("data/model/lb.joblib")

    array = np.array([[
                     32,
                     "Private",
                     "Some-college",
                     "Married-civ-spouse",
                     "Exec-managerial",
                     "Husband",
                     "Black",
                     "Male",
                     80,
                     "United-States"
                     ]])
    df_temp = DataFrame(data=array, columns=[
        "age",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
    ])

    x, _, _, _ = src.common_functions.process_data(
                df_temp,
                categorical_features=src.common_functions.get_cat_features(),
                encoder=encoder, lb=lb, training=False)
    pred = src.common_functions.inference(model, x)
    y = lb.inverse_transform(pred)[0]
    assert y == ">50K"