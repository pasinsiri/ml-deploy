import numpy as np
import logging
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
        data,
        cat_cols: list,
        training: bool,
        label: str = None,
        encoder=None,
        lb=None):
    # Create x and y dataframes
    x = data.copy()
    if label is not None:
        y = x.pop(label).values
    else:
        y = np.array([])

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
        y = lb.fit_transform(y).ravel()
    else:
        x_cat = encoder.transform(x_cat)
        try:
            y = lb.fit_transform(y).ravel()
        except (ValueError, AttributeError):
            logging.info(
                'y is not passed. it is ignored  \
                    since training is set to False')

    x = np.concatenate([x_cat, x_num], axis=1)
    return x, y, encoder, lb
