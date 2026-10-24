# MediSafe: Prescription Safety Checker

Checks whether a prescribed medicine is safe for a patient using allergies, current medicines,
medical conditions (15 conditions incl. pregnancy, diabetes, asthma, kidney/liver disease...) and age.
56 medicines across 17 groups. Machine-learning model + drug-safety rule engine. Educational use only.

## Files
| File | Purpose |
|---|---|
| app.py | Streamlit website |
| knowledge.py | Medicines, conditions, allergies, interaction rules |
| model.pkl | Trained model |
| medicine_dataset.csv | 12,000-patient synthetic dataset |
| patient_records.csv | Saved checks (written by the app) |
| train_model.py | Regenerates dataset + model (`python train_model.py`) |
| requirements.txt | Dependencies |
| .streamlit/config.toml | Theme |

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Deploy free on Streamlit Community Cloud
1. Upload ALL files (including the `.streamlit` folder, with `requirements.txt` in the repo root) to a GitHub repo.
2. Go to https://share.streamlit.io, choose New app, pick the repo and `app.py`, then Deploy.
