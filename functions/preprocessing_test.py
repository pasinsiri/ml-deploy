"""
Test script for preprocessing
"""
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import pytest
from joblib import load
from functions.preprocessing import process_data

@pytest.fixture
def data():
    """
    Get dataset
    """
    df = pd.read_csv('./data/census.csv')
    return df


def test_process_data(data):
    """
    Check split have same number of rows for x and y after processing
    """
    encoder = load('model/encoder.joblib')
    lb = load('/model/lb.joblib')

    x_test, y_test, _, _ = process_data(
        data,
        categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ],
        label="salary", encoder=encoder, lb=lb, training=False)
    assert len(x_test) == len(y_test)


def test_process_encoder(data):
    """
    Check consistency of process_data in parts of encoder and label binarizer
    """
    encoder_test = load("data/model/encoder.joblib")
    lb_test = load("data/model/lb.joblib")

    _, _, encoder, lb = process_data(
        data,
        categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ],
        label="salary", training=True)

    _, _, _, _ = process_data(
        data,
        categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ],
        label="salary", encoder=encoder_test, lb=lb_test, training=False)

    assert encoder.get_params() == encoder_test.get_params()
    assert lb.get_params() == lb_test.get_params()


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


def test_inference_below():
    """
    Check inference performance
    """
    model = load("data/model/model.joblib")
    encoder = load("data/model/encoder.joblib")
    lb = load("data/model/lb.joblib")

    array = np.array([[
                     19,
                     "Private",
                     "HS-grad",
                     "Never-married",
                     "Own-child",
                     "Husband",
                     "Black",
                     "Male",
                     40,
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
    assert y == "<=50K"