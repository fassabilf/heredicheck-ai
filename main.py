import random

# === 5. Endpoint untuk Prediksi Probabilitas ===
@app.post("/predict_proba")
def predict_disease_proba(data: PredictionRequest):
    target_diseases = ["Diabetes", "Hypertension", "Cancer", "Heart Disease", "Alzheimer", "Asthma"]

    result = {}
    for p in data.patients:
        result[p.id] = {
            disease: round(random.uniform(0.0, 1.0), 4)  # nilai antara 0.0000 - 1.0000
            for disease in target_diseases
        }

    return {"probabilities": result}
