# **Heredicheck-AI: Graph-Based Predictive AI for Hereditary Disease Risk Assessment**

## **Overview**
Heredicheck-AI is an **AI-powered system** designed to **predict hereditary disease risks** by analyzing **structured patient relationships and medical histories** using **Graph Neural Networks (GNNs)**. The project leverages **FHIR-compliant synthetic data generation** and **graph-based learning** to model **genetic inheritance patterns**, allowing for **early detection of hereditary diseases**.

This project was developed as part of the **MeldRx Predictive AI App Challenge** and provides an **end-to-end pipeline** for:

✔ **Generating FHIR-compliant synthetic patient data**  
✔ **Linking family relationships and inherited diseases**  
✔ **Training a GNN for multi-label hereditary disease prediction**  
✔ **Deploying a FastAPI-based inference system**  

---

## **1. Step-by-Step: Generating Synthetic Patient Data with Synthea**
### **1.1 Cloning and Running Synthea**
To generate realistic **FHIR-compatible patient data**, we use **[Synthea](https://github.com/synthetichealth/synthea)**, a synthetic patient generator that follows **real-world epidemiological distributions**.

#### **Step 1: Clone Synthea**
```bash
git clone https://github.com/synthetichealth/synthea.git
cd synthea
```

#### **Step 2: Configure Synthea (Optional)**
Modify `synthea.properties` to **adjust parameters** (e.g., number of patients, disease prevalence).

#### **Step 3: Generate 1000 Synthetic Patients**
```bash
./run_synthea -p 1000
```
This command generates **1000 patients** with structured **FHIR-based health records**.

---

### **1.2 Addressing Missing Relationships in Synthea's Output**
#### **Problem: Missing `RelatedPerson` and `FamilyMemberHistory` Links**
By default, **Synthea does not include**:
1. **Direct links between patients in `RelatedPerson` records** (i.e., patients do not reference their family members explicitly).
2. **Inheritance relationships in `FamilyMemberHistory` (FMH)**.

#### **Solution: Post-Processing to Link Family Members**
To **correct this**, we developed a **post-processing pipeline** that:

✔ **Links `RelatedPerson` records to actual patient IDs.**  
✔ **Assigns hereditary diseases based on epidemiological probability.**  
✔ **Ensures relationships are **bi-directional** (e.g., a patient’s father should also reference them as a child).  

This ensures **realistic family relationships**, which are **critical for graph-based AI models**.

---

## **2. Dataset Generation: Structuring Patient Data for AI Models**
### **2.1 Probability-Based Disease Inheritance**
Each synthetic patient’s **family history and inherited diseases** are assigned based on **real-world epidemiological distributions**:

| Disease        | Probability (%) |
|---------------|---------------|
| Diabetes      | 12.5%          |
| Hypertension  | 35.0%          |
| Cancer        | 7.5%           |
| Heart Disease | 22.5%          |
| Alzheimer’s   | 3.0%           |
| Asthma        | 10.0%          |

Using these probabilities, **diseases propagate within family trees**, mimicking **real-world hereditary transmission patterns**.

### **2.2 Constructing Family Relationships**

✔ **Assign parents (`Father`, `Mother`) probabilistically**  
✔ **Generate siblings (`Brother`, `Sister`) with variable family sizes**  
✔ **Ensure relationships remain **consistent and bi-directional**  

Each patient is linked to their **biological relatives**, ensuring that the **graph representation captures real-world genetic relationships**.

---

## **3. Graph-Based AI Model: Learning from Structured Relationships**
### **3.1 Why Use Graph Neural Networks (GNNs)?**
Traditional ML models **struggle to capture family relationships**. **Heredicheck-AI solves this problem** by using **Graph Neural Networks (GNNs)**, which allow:

✔ **Multi-generation disease modeling**  
✔ **Inference of genetic disease risk beyond immediate relatives**  
✔ **Graph-based learning to capture hidden relationships**  

### **3.2 Graph Representation**
- **Nodes**: Each patient is a **node**, with features derived from **medical history embeddings (TF-IDF vectorization)**.
- **Edges**: Connections (`Father`, `Mother`, `Sibling`) define **relationships**, forming a **structured knowledge graph**.
- **Features**: Each node contains a **vectorized representation of medical history**, allowing **disease prediction from patient context**.

### **3.3 Multi-Label Disease Prediction**

✔ **Simultaneous prediction of 6 hereditary diseases**  
✔ **Hierarchical learning from genetic patterns**  
✔ **Scalability to additional diseases in future versions**  

---

## **4. Training the GNN Model**
### **4.1 Model Training Pipeline**
✔ **80% Training / 20% Validation Split** using **iterative stratification**  
✔ **Graph Construction with PyTorch Geometric (PyG)**  
✔ **Multi-label classification using GCNConv layers**  

### **4.2 Model Architecture**
- **Graph Convolutional Networks (GCN)**:
  - **Layer 1**: Encodes **patient node features** using relational graph convolution.
  - **ReLU Activation**: Introduces non-linearity for **better feature representation**.
  - **Layer 2**: Outputs **final disease probabilities** for each patient.

🔹 **Loss Function**: Binary Cross-Entropy with Logits (BCEWithLogitsLoss) for **multi-label classification**  
🔹 **Optimizer**: Adam optimizer with **learning rate = 0.01**  

---

## **5. API Deployment for Real-Time Disease Risk Prediction**
### **5.1 Model Saving & Vectorizer Persistence**
After training, **Heredicheck-AI exports its GNN model and vectorizer** for real-time inference:
```python
torch.save(model.state_dict(), "gnn_model_weights.pt")
with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)
```
These files enable **instant deployment** of the trained model.

### **5.2 FastAPI-Powered Prediction API**
A **FastAPI backend** enables **real-time inference**, integrating into **FHIR-compliant healthcare systems**.

#### **Example API Request**
```json
{
  "patients": [
    {"id": "A1", "patient_condition": "diabetes hypertension"},
    {"id": "B2", "patient_condition": "cancer heart disease"},
    {"id": "C3", "patient_condition": "asthma"}
  ],
  "relationships": [
    {"id_patient": "A1", "related_patient": "B2", "type": "Father"},
    {"id_patient": "B2", "related_patient": "C3", "type": "Brother"}
  ]
}
```

#### **Example Response**
```json
{
  "probabilities": {
    "A1": {"Diabetes": 0.82, "Hypertension": 0.76, "Cancer": 0.31, "Heart Disease": 0.55, "Alzheimer": 0.12, "Asthma": 0.48},
    "B2": {"Diabetes": 0.41, "Hypertension": 0.88, "Cancer": 0.67, "Heart Disease": 0.39, "Alzheimer": 0.09, "Asthma": 0.21},
    "C3": {"Diabetes": 0.19, "Hypertension": 0.52, "Cancer": 0.22, "Heart Disease": 0.48, "Alzheimer": 0.07, "Asthma": 0.91}
  }
}
```

---

## **6. Installation & Usage**
```bash
pip install torch torch-geometric scikit-learn fastapi hypercorn numpy pandas
hypercorn main:app --reload
```

---

## **Conclusion**
Heredicheck-AI is an **advanced AI-driven system** designed to **enhance hereditary disease risk prediction** through **graph-based learning**. By integrating **synthetic patient data, probabilistic disease modeling, and GNNs**, we provide **a powerful tool for predictive healthcare analytics**.

This AI system **can be expanded** to include **additional diseases, larger datasets, and real-world clinical applications**.  

We welcome contributions to **improve and scale this project further**.