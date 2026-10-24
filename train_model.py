# Regenerates medicine_dataset.csv and model.pkl.  Run:  python train_model.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from knowledge import generate_dataset, build_features, row_to_patient

df = generate_dataset(n=12000)
df.to_csv("medicine_dataset.csv", index=False)
print("Dataset:", df.shape, "| safe share:", round(df["is_safe"].mean(), 3))

X = pd.DataFrame([build_features(row_to_patient(r), r["prescribed_medicine"]) for _, r in df.iterrows()])
y = df["is_safe"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced",
                               random_state=42, n_jobs=-1).fit(X_tr, y_tr)
print(classification_report(y_te, model.predict(X_te), target_names=["Unsafe", "Safe"]))

final = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced",
                               random_state=42, n_jobs=-1).fit(X, y)
joblib.dump({"model": final, "columns": list(X.columns)}, "model.pkl", compress=3)
print("Saved model.pkl and medicine_dataset.csv")
