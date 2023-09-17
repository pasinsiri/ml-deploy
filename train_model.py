# Script to train machine learning model.

import pandas as pd
import numpy as np
import logging
import joblib
from functions.preprocessing import process_data

from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

# Add code to load in the data.
train_data = pd.read_csv('./data/census_income/adult.data', header=None)
test_data = pd.read_csv('./data/census_income/adult.test', header=None, skiprows=1)

# Get column names
meta = pd.read_table('./data/census_income/adult.names', header=None)
col_list = meta.iloc[93:][0].apply(lambda x: x.split(':')[0]).values.tolist()
col_list.append('salary')
train_data.columns = col_list
test_data.columns = col_list


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
    train_data, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test_data, categorical_features=cat_features, label="salary", training=False
)

# Train and save a model.
cv = KFold(n_splits=10, shuffle=True, random_state=42)
clf = GradientBoostingClassifier(n_estimators=50)
clf.fit(X_train, y_train)
scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=cv)
logging.info(f'Average accuracy from cross-validation on training data = {np.mean(scores)}')
joblib.dump(clf, 'model/model.joblib')
joblib.dump(encoder, 'model/encoder.joblib')
joblib.dump(lb, 'model/lb.joblib')

# Evaluate model on test data