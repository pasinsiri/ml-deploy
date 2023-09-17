import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

def process_data(data, cat_cols: list, label: str, training: bool, encoder = None, lb = None):
    # Create x and y dataframes
    x = data.copy()
    y = x.pop(label)

    # Split categorical / numerical columns
    x_cat = x[cat_cols]
    x_num = x.drop(*[cat_cols], axis=1)
    
    # Trim string columns
    for c in x_cat.columns:
        x_cat[c] = x_cat[c].str.strip()

    if training:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        x_cat = encoder.fit_transform(x_cat)

        lb = LabelBinarizer()
        y = lb.fit_transform(y.values).ravel()
    else:
        x_cat = encoder.transform(x_cat)
        y = lb.transform(y.values).ravel()

    x = np.concatenate([x_cat, x_num], axis=1)
    return x, y, encoder, lb
