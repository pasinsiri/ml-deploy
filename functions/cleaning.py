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

    # clean categorical columns
    cat_cols = get_categorical_columns()
    for c in cat_cols:
        df[c] = df[c].str.strip()
        df[c] = df[c].replace('?', None)

    return df.dropna()
