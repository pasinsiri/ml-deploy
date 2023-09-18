import pandas as pd
import joblib
import pytest
from .cleaning import get_categorical_columns
from .preprocessing import process_data
from .model_evaluation import inference


@pytest.fixture
def data():
    """
    Load dataset
    """
    df = pd.read_csv('data/census.csv')
    return df


def test_inference_above():
    """
    Check inference performance for case salary >50k
    """
    model = joblib.load('model/model.joblib')
    encoder = joblib.load('model/encoder.joblib')
    lb = joblib.load('/model/lb.joblib')

    test_case_above = [
        40,
        'Private',
        193524,
        'Doctorate',
        16,
        'Married-civ-spouse',
        'Prof-specialty',
        'Husband',
        'White',
        'Male',
        0,
        0,
        60,
        'United-States']
    test_case_above_df = pd.DataFrame(
        [test_case_above],
        columns=[
            'age',
            'workclass',
            'fnlgt',
            'education',
            'education-num',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital-gain',
            'capital-loss',
            'hours-per-week',
            'native-country'])

    x, _, _, _ = process_data(
        test_case_above_df,
        cat_cols=get_categorical_columns(),
        encoder=encoder,
        lb=lb,
        training=False
    )
    pred = inference(model, x)
    y = lb.inverse_transform(pred)[0]
    assert y == ">50K"


def test_inference_below():
    """
    Check inference performance for case salary <=50K
    """
    model = joblib.load('model/model.joblib')
    encoder = joblib.load('model/encoder.joblib')
    lb = joblib.load('/model/lb.joblib')

    test_case_below = [
        31,
        'Private',
        507875,
        '9th',
        5,
        'Married-civ-spouse',
        'Machine-op-inspct',
        'Husband',
        'White',
        'Male',
        0,
        0,
        43,
        'United-States']
    test_case_below_df = pd.DataFrame(
        [test_case_below],
        columns=[
            'age',
            'workclass',
            'fnlgt',
            'education',
            'education-num',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital-gain',
            'capital-loss',
            'hours-per-week',
            'native-country'])

    x, _, _, _ = process_data(
        test_case_below_df,
        cat_cols=get_categorical_columns(),
        encoder=encoder,
        lb=lb, training=False
    )
    pred = inference(model, x)
    y = lb.inverse_transform(pred)[0]
    assert y == "<=50K"
