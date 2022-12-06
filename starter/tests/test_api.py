import json
from fastapi.testclient import TestClient
from main import app

testApp = TestClient(app)

testPayload = {
  "age": 29,
  "workclass": "State-gov",
  "fnlgt": 77516,
  "education": "Bachelors",
  "education_num": 13,
  "marital_status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital_gain": 2174,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": "United-States"
}

badPayload = {}

def test_get():
    res = testApp.get("/")
    assert res.json()["result"] == "Welcome to CENSUS API. You can type in a JSON body containing 14 attributes to get back a salary prediction"
    assert res.status_code == 200

def test_post_correct(testPayload):
    res = testApp.post("/predict")
    assert res.status_code == 200
    assert res.json()["prediction"] == "Income < 50k"

def test_post_wrong(badPayload):
    res = testApp.post("/predict")
    assert res.status_code != 200
