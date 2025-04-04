from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import random
from fastapi.middleware.cors import CORSMiddleware

# === Inisialisasi FastAPI ===
app = FastAPI(title="HerediCheck-AI (Mocked)", version="1.0")

# === Setup CORS ===
origins = [
    "http://localhost:3000",
    "https://heredicheck.vercel.app",
    "https://api-heredicheck.up.railway.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Define Input Schema ===
class Patient(BaseModel):
    id: str
    patient_condition: str
    patient_immunization: str
    patient_allergy: str

class Relationship(BaseModel):
    id_patient: str
    related_patient: str
    type: str  # Relationship type

class PredictionRequest(BaseModel):
    patients: List[Patient]
    relationships: List[Relationship]

# === Daftar Penyakit Target ===
target_diseases = [
    "Diabetes",
    "Hypertension",
    "Cancer",
    "Heart Disease",
    "Alzheimer",
    "Asthma"
]

# === Endpoint Prediksi dengan Probabilitas Acak ===
@app.post("/predict_proba")
def predict_disease_proba(data: PredictionRequest):
    result = {}
    for patient in data.patients:
        result[patient.id] = {
            disease: round(random.uniform(0.0, 1.0), 4)
            for disease in target_diseases
        }

    return {"probabilities": result}
