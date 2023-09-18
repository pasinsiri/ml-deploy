import pandas as pd
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from .cleaning import get_categorical_columns
from .preprocessing import process_data


def calculate_f1(y_true, y_pred):
    """validate the model using F1 score

    Args:
        y_true: known labels (binarized)
        y_pred: predicted labels (binarized)

    Returns:
        float: F1 score
    """
    f1 = f1_score(y_true, y_pred)
    return f1


def inference(model, x):
    """run model inference and return predicted values

    Args:
        model: an ML model
        x: input data to be predicted

    Returns:
        np.array: predicted values
    """

    y_preds = model.predict(x)
    return y_preds


def slicing():
    """
    slice the data by features
    """
    df = pd.read_csv("data/prepared/census.csv")
    _, test = train_test_split(df, test_size=0.20)

    trained_model = joblib.load("data/model/model.joblib")
    encoder = joblib.load("data/model/encoder.joblib")
    lb = joblib.load("data/model/lb.joblib")

    cat_cols = get_categorical_columns()
    slice_values = []

    for cat in cat_cols:
        for cls in test[cat].unique():
            sliced_df = test[test[cat] == cls]
            x_test, y_test, _, _ = process_data(
                sliced_df, categorical_features=cat_cols, label="salary", encoder=encoder, lb=lb, training=False)

            y_pred = trained_model.predict(x_test)
            f1 = calculate_f1(y_test, y_pred)
            line = f'[{cat} = {cls}]: F1 Score = {f1}'
            logging.info(line)
            slice_values.append(line)

    with open('model/slice_output.txt', 'w') as out:
        for slice_value in slice_values:
            out.write(slice_value + '\n')
