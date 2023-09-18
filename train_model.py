# Script to train machine learning model.

import pandas as pd
import numpy as np
import logging
import joblib
from functions.cleaning import get_categorical_columns, clean_data
from functions.preprocessing import process_data

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

# Add code to load in the data.
raw_df = pd.read_csv('./data/census.csv', skipinitialspace=True)

cat_features = get_categorical_columns()
df = clean_data(raw_df)

x_train, y_train, encoder, lb = process_data(
    df, cat_cols=cat_features, label="salary", training=True
)

# Train and save a model.
cv = KFold(n_splits=10, shuffle=True, random_state=42)
clf = GradientBoostingClassifier(n_estimators=50)
clf.fit(x_train, y_train)
scores = cross_val_score(clf, x_train, y_train, scoring='accuracy', cv=cv)
logging.info(
    f'Average accuracy from cross-validation on training data = {np.mean(scores)}')
joblib.dump(clf, 'model/model.joblib')
joblib.dump(encoder, 'model/encoder.joblib')
joblib.dump(lb, 'model/lb.joblib')
