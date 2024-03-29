# Script to train machine learning model.

import pandas as pd
import numpy as np
import logging
import joblib
from functions.cleaning import get_categorical_columns, clean_data
from functions.preprocessing import process_data
from functions.model_evaluation import cv_score, slicing

from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier

# Add code to load in the data.
raw_df = pd.read_csv('data/census.csv', skipinitialspace=True)

cat_features = get_categorical_columns()
df = clean_data(raw_df)

x_train, y_train, encoder, lb = process_data(
    df, cat_cols=cat_features, label="salary", training=True
)

# Train and save a model.
cv = KFold(n_splits=5, shuffle=True, random_state=42)
clf = GradientBoostingClassifier(n_estimators=50)
clf.fit(x_train, y_train)
scores = cv_score(clf, x_train, y_train, cv)
logging.info(
    f'Average F1 score from cross-validation  \
        on training data = {np.mean(scores)}')
joblib.dump(clf, 'model/model.joblib')
joblib.dump(encoder, 'model/encoder.joblib')
joblib.dump(lb, 'model/lb.joblib')

# slice the data
slicing()
