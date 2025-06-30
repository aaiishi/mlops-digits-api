from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Charger le modèle
model = joblib.load("model.joblib")

# Initialiser l'app FastAPI
app = FastAPI()

# Définir le format d'entrée
class InputData(BaseModel):
    data: list[list[float]]

@app.get("/")
def root():
    return {"message": "API OK"}

@app.post("/predict")
def predict(input_data: InputData):
    data = np.array(input_data.data)
    prediction = model.predict(data).tolist()
    return {"prediction": prediction[0]}
