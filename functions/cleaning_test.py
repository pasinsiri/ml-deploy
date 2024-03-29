import pandas as pd
import pytest
from .cleaning import clean_data


@pytest.fixture
def data():
    """
    Load dataset
    """
    df = pd.read_csv('data/census.csv', skipinitialspace=True)
    df = clean_data(df)
    return df


def test_null(data):
    """
    Cleaned data is assumed to have no null values
    """
    assert data.shape == data.dropna().shape


def test_question_mark(data):
    """
    Cleaned data is assumed to have no question marks value
    """
    assert '?' not in data.values
