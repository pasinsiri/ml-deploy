"""
Clean values
"""


def get_categorical_columns():
    cat_cols = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return cat_cols


def clean_data(raw_df):
    df = raw_df.copy()

    # trim column names
    df.columns = [c.strip() for c in df.columns]

    # clean categorical columns
    cat_cols = get_categorical_columns()
    for c in cat_cols:
        df[c] = df[c].replace('?', None)

    return df.dropna()
