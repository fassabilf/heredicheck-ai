from fastapi import FastAPI
import torch
import torch.nn.functional as F
import re
import numpy as np
from pydantic import BaseModel
from typing import List
from torch_geometric.data import Data
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# === 1. Load Model GNN dan TF-IDF Vectorizer ===
from model import GNNModel  # Pastikan file model.py berisi definisi arsitektur GNN

# Inisialisasi ulang model GNN
hidden_channels = 64
out_channels = 6  # Jumlah penyakit target
# Load TF-IDF Vectorizer
with open("tfidf.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

input_dim = len(tfidf_vectorizer.get_feature_names_out())  # Pastikan jumlah fitur cocok
model = GNNModel(in_channels=input_dim, hidden_channels=64, out_channels=6)
model.load_state_dict(torch.load("gnn_model_weights.pt"))


model.load_state_dict(torch.load("gnn_model_weights.pt"))
model.eval()  # Set model ke mode evaluasi

# === 2. Inisialisasi FastAPI ===
app = FastAPI(title="Medical Predictive API with GNN", version="1.0")

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
class Patient(BaseModel):
    id: str
    patient_condition: str

class Relationship(BaseModel):
    id_patient: str
    related_patient: str
    type: str  # Relationship type (Father, Mother, Brother, Sister, etc.)

class PredictionRequest(BaseModel):
    patients: List[Patient]
    relationships: List[Relationship]

# === 5. Endpoint untuk Prediksi Probabilitas ===
@app.post("/predict_proba")
def predict_disease_proba(data: PredictionRequest):
    # 1. Buat dictionary untuk mapping ID pasien ke index numerik
    patient_ids = {p.id: idx for idx, p in enumerate(data.patients)}
    
    # 2. Proses kondisi pasien
    patient_texts = [clean_text(p.patient_condition) for p in data.patients]
    X_tfidf = tfidf_vectorizer.transform(patient_texts).toarray()

    # 3. Konversi ke tensor untuk node features
    node_features = torch.tensor(X_tfidf, dtype=torch.float32)

    # 4. Proses relationships sebagai edge index
    edge_index = []
    for rel in data.relationships:
        if rel.id_patient in patient_ids and rel.related_patient in patient_ids:
            edge_index.append([patient_ids[rel.id_patient], patient_ids[rel.related_patient]])

    # 5. Konversi edge ke tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).T if edge_index else torch.tensor([], dtype=torch.long).view(2, -1)

    # 6. Buat graph untuk GNN
    input_graph = Data(x=node_features, edge_index=edge_index)

    # 7. Prediksi dengan GNN
    with torch.no_grad():
        logits = model(input_graph)
        y_proba = torch.sigmoid(logits).numpy()  # Konversi ke probabilitas

    # 9. Konversi hasil probabilitas ke bentuk JSON
    result = {}
    target_diseases = ["Diabetes", "Hypertension", "Cancer", "Heart Disease", "Alzheimer", "Asthma"]

    for i, p in enumerate(data.patients):
        result[p.id] = {target_diseases[j]: float(y_proba[i, j]) for j in range(len(target_diseases))}  # Convert ke float biasa

    return {"probabilities": result}

