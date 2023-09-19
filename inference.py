import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union, Optional

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
async def say_hello():
    return {"greeting": "Hello World!"}

# load model artifacts on startup of the application to reduce latency


@app.on_event("startup")
async def startup_event():
    model = joblib.load('model/model.joblib')
    encoder = joblib.load('model/encoder.joblib')
    lb = joblib.load('model/lb.joblib')
