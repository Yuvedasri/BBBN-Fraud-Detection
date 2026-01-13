# BBBN-Fraud-Detection

**Behavior–Belief–Boosted Fraud Network (BBBN)** — A complete end-to-end system for detecting healthcare provider fraud using behavioral, anomaly, and network intelligence.

---

## System Overview

BBBN detects fraudulent providers by combining:

✔ Feature engineering from claims  
✔ Isolation Forest (anomaly detection)  
✔ LSTM behavioral modeling  
✔ XGBoost fusion model  
✔ Rule-based decision agent  
✔ Interactive Streamlit dashboard

---

## Project Structure
backend/ – model training & pipeline scripts
data/raw/ – raw claims CSVs (not included)
data/processed/ – generated features & results (not included)
ui/app.py – Streamlit dashboard
requirements.txt – dependencies


---

## 🛠 How to Run

1. **Clone the repo**
    ```bash
    git clone https://github.com/Yuvedasri/BBBN-Fraud-Detection.git
    cd BBBN-Fraud-Detection
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run models**
    ```bash
    python backend/ingestion.py
    python backend/feature_engineering.py
    python backend/anomaly_model.py
    python backend/behavior_model.py
    python backend/fusion_model.py
    python backend/agent.py
    ```

4. **Launch the dashboard**
    ```bash
    python -m streamlit run ui/app.py
    ```

---

## Dashboard

✔ View provider risk  
✔ Explore decisions  
✔ Drill down into fraud scores  

---

## Results

✔ High ROC-AUC (>0.93)  
✔ Explainable decisions (Approve / Flag / Block)  
✔ Modular & extendable

---

## License

MIT License

