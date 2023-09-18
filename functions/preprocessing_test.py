"""
Test script for preprocessing
"""
import pandas as pd
import numpy as np
import pytest
from joblib import load
from .preprocessing import process_data

@pytest.fixture
def data():
    """
    Load dataset
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