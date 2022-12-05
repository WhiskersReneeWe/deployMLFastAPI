from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Predictor(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    tax: Union[float, None] = None

@app.post("/predict")
async def predict(payload: Predictor):
    pass

@app.get("/")
async def welcome():
    result = "Welcome to CENSUS API. " \
             "You can type in a JSON body containing 14 attributes to get back a salary prediction"

    return {"result": result, "health_check": "OK"}
