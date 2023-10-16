import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from functions.preprocessing import process_data

# instantiate FastAPI app
app = FastAPI(title="Inference API",
              description="An API that takes a sample and runs an inference",
              version="1.0.0")


class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


@app.get("/")
async def greetings():
    return "Hello! This is a machine learning model API"


@app.on_event("startup")
async def startup_event():
    global model, encoder, lb
    model = joblib.load('model/model.joblib')
    encoder = joblib.load('model/encoder.joblib')
    lb = joblib.load('model/lb.joblib')


@app.post("/inference")
async def ingest_data(inference: InputData):
    data = {
        'age': inference.age,
        'workclass': inference.workclass,
        'fnlgt': inference.fnlgt,
        'education': inference.education,
        'education-num': inference.education_num,
        'marital-status': inference.marital_status,
        'occupation': inference.occupation,
        'relationship': inference.relationship,
        'race': inference.race,
        'sex': inference.sex,
        'capital-gain': inference.capital_gain,
        'capital-loss': inference.capital_loss,
        'hours-per-week': inference.hours_per_week,
        'native-country': inference.native_country
    }

    # prepare the sample for inference as a dataframe
    sample = pd.DataFrame(data, index=[0])

    # apply transformation to sample data
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

    model = joblib.load('model/model.joblib')
    encoder = joblib.load('model/encoder.joblib')
    lb = joblib.load('model/lb.joblib')

    sample, _, _, _ = process_data(
        sample,
        cat_cols=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # get model prediction which is a one-dim array like [1]
    prediction = model.predict(sample)

    # convert prediction to label and add to data output
    if prediction[0] > 0.5:
        prediction = '>50K'
    else:
        prediction = '<=50K',
    data['prediction'] = prediction
    return data

if __name__ == '__main__':
    pass
