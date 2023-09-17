# Script to train machine learning model.

import pandas as pd
from sklearn.model_selection import train_test_split
from functions.preprocessing import process_data

# Add code to load in the data.
data = pd.read_csv('./data/census_income/adult.data', header=None)
test_data = pd.read_csv('./data/census_income/adult.test', header=None, skiprows=1)

# Get column names
meta = pd.read_table('./data/census_income/adult.names', header=None)
col_list = meta.iloc[93:][0].apply(lambda x: x.split(':')[0]).values.tolist()
col_list.append('salary')
data.columns = col_list
test_data.columns = col_list

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.

# Train and save a model.