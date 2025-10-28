from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load model
with open("pipeline_v1.bin", "rb") as f:
    model = pickle.load(f)

# Define request schema
class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Model is running!"}

@app.post("/predict")
def predict(client: Client):
    data = client.dict()
    X = [data]
    proba = model.predict_proba(X)[0, 1]
    return {"conversion_probability": proba}
