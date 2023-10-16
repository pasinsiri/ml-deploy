"""
POST live API
"""

import requests

url = 'https://ml-deployment-fastapi.onrender.com/inference/'

sample = {
    'age': 50,
    'workclass': "Private",
    'fnlgt': 234721,
    'education': "Doctorate",
    'education_num': 16,
    'marital_status': "Separated",
    'occupation': "Exec-managerial",
    'relationship': "Not-in-family",
    'race': "Black",
    'sex': "Female",
    'capital_gain': 0,
    'capital_loss': 0,
    'hours_per_week': 50,
    'native_country': "United-States"
}

# post a request