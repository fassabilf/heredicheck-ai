from fastapi import FastAPI
import pickle
import re
import numpy as np
from scipy.sparse import hstack
from pydantic import BaseModel

# === 1. Load Model Logistic Regression dan TF-IDF Vectorizer ===
with open("logreg.pkl", "rb") as f:
    logreg = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# === 2. Inisialisasi FastAPI ===
app = FastAPI(title="Medical Predictive API", version="1.0")

# === 3. Preprocessing Function ===
def clean_text(text):
    if text is None or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === 4. Define Input Schema ===
class PatientData(BaseModel):
    patient_conditions_text_cleaned: str
    father_merged_condition: str
    mother_merged_condition: str
    brother_merged_condition: str
    sister_merged_condition: str

# === 5. Endpoint untuk Prediksi Probabilitas ===
@app.post("/predict_proba")
def predict_disease_proba(data: PatientData):
    # Preprocessing input
    input_texts = [
        clean_text(data.patient_conditions_text_cleaned),
        clean_text(data.father_merged_condition),
        clean_text(data.mother_merged_condition),
        clean_text(data.brother_merged_condition),
        clean_text(data.sister_merged_condition)
    ]

    # Transform input menggunakan TF-IDF
    X_input_features = [tfidf_vectorizer.transform([text]) for text in input_texts]
    X_combined = hstack(X_input_features)

    # Lakukan prediksi probabilitas
    y_proba = logreg.predict_proba(X_combined)

    # Label target yang digunakan saat training
    target_labels = ["Diabetes", "Hypertension", "Cancer", "Heart Disease", "Alzheimer", "Asthma"]

    # Konversi hasil probabilitas ke bentuk JSON
    result = {target_labels[i]: round(y_proba[0, i], 4) for i in range(len(target_labels))}

    return {"probabilities": result}
