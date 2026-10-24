import os
import uuid
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

from knowledge import DRUGS, ALLERGY_OPTIONS, check_safety, build_features, row_to_patient

st.set_page_config(page_title="MediSafe AI", page_icon="💊", layout="wide")

RECORDS_FILE = "patient_records.csv"
DATASET_FILE = "medicine_dataset.csv"
RECORD_COLS = [
    "record_id", "timestamp", "patient_name", "doctor_name", "age", "gender", "pregnant",
    "kidney_disease", "liver_disease", "allergies", "previous_medicines",
    "prescribed_medicine", "diagnosis_notes", "verdict", "ml_prob_safe", "reasons",
]

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
.hero{padding:1.6rem 2rem;border-radius:18px;margin-bottom:1rem;
      background:linear-gradient(120deg,#0ea5e9 0%,#6366f1 55%,#a855f7 100%);}
.hero h1,.hero p{color:#fff !important;margin:0;}
.hero h1{font-size:2.3rem;padding:0 0 .3rem 0;}
.result{padding:1.1rem 1.4rem;border-radius:14px;font-size:1.25rem;font-weight:700;margin:.6rem 0;}
.ok{background:rgba(34,197,94,.15);border:1.5px solid #22c55e;}
.bad{background:rgba(239,68,68,.15);border:1.5px solid #ef4444;}
.warn{background:rgba(245,158,11,.15);border:1.5px solid #f59e0b;}
.chip{display:inline-block;padding:3px 12px;border-radius:999px;margin:2px 4px 2px 0;
      background:rgba(99,102,241,.18);font-size:.85rem;}
.reason{padding:.5rem .8rem;border-left:4px solid #ef4444;background:rgba(239,68,68,.08);
        border-radius:6px;margin:.35rem 0;}
.bar{height:14px;border-radius:999px;background:rgba(148,163,184,.3);overflow:hidden;}
.bar>div{height:100%;border-radius:999px;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- data / model
@st.cache_resource
def load_model():
    try:
        b = joblib.load("model.pkl")
        return b["model"], b["columns"]
    except Exception:
        # Fallback: retrain if model.pkl is missing or built with another scikit-learn version
        d = pd.read_csv(DATASET_FILE)
        X = pd.DataFrame([build_features(row_to_patient(r), r["prescribed_medicine"]) for _, r in d.iterrows()])
        m = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
        m.fit(X, d["is_safe"])
        return m, list(X.columns)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_FILE)


def load_records():
    try:
        if os.path.exists(RECORDS_FILE) and os.path.getsize(RECORDS_FILE) > 0:
            return pd.read_csv(RECORDS_FILE)
    except Exception:
        pass
    return pd.DataFrame(columns=RECORD_COLS)


def save_record(rec):
    df = pd.concat([load_records(), pd.DataFrame([rec])], ignore_index=True)
    df.to_csv(RECORDS_FILE, index=False)


model, columns = load_model()
data = load_dataset()

# ---------------------------------------------------------------- session defaults
DEFAULTS = dict(p_name="", p_doctor="", p_age=35, p_gender="Male", p_pregnant=False,
                p_kidney=False, p_liver=False, p_allergies=[], p_prev=[],
                p_rx=sorted(DRUGS)[0], p_notes="")
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

EXAMPLES = {
    "🤧 Penicillin allergy": dict(p_name="Demo: Anita", p_age=28, p_gender="Female", p_pregnant=False,
                                  p_kidney=False, p_liver=False, p_allergies=["penicillin"],
                                  p_prev=["Cetirizine"], p_rx="Amoxicillin"),
    "🩸 On Warfarin": dict(p_name="Demo: Ravi", p_age=58, p_gender="Male", p_pregnant=False,
                          p_kidney=False, p_liver=False, p_allergies=[],
                          p_prev=["Warfarin", "Atorvastatin"], p_rx="Ibuprofen"),
    "🤰 Pregnant patient": dict(p_name="Demo: Sneha", p_age=26, p_gender="Female", p_pregnant=True,
                               p_kidney=False, p_liver=False, p_allergies=[], p_prev=[], p_rx="Doxycycline"),
    "✅ Healthy adult": dict(p_name="Demo: Arjun", p_age=32, p_gender="Male", p_pregnant=False,
                            p_kidney=False, p_liver=False, p_allergies=[], p_prev=[], p_rx="Paracetamol"),
}


def load_example(key):
    st.session_state.update(EXAMPLES[key])


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 💊 MediSafe AI")
    st.caption("Prescription safety checker")
    st.markdown("**Try an example**")
    for label in EXAMPLES:
        st.button(label, on_click=load_example, args=(label,), width="stretch")
    st.divider()
    st.caption("⚠️ Educational demo with synthetic data and simplified rules. "
               "Not for real clinical decisions.")

# ---------------------------------------------------------------- hero + stats
st.markdown(
    """
<div class="hero">
  <h1>💊 MediSafe AI</h1>
  <p>Is the prescribed medicine right for this patient? Checks allergies, previous medicines,
  pregnancy, organ conditions and age, using machine learning plus a drug-safety rule engine.</p>
</div>
""",
    unsafe_allow_html=True,
)
records = load_records()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Patients in dataset", f"{len(data):,}")
m2.metric("Medicines covered", len(DRUGS))
m3.metric("Unsafe cases in dataset", f"{(data['is_safe'] == 0).mean() * 100:.0f}%")
m4.metric("Patient records saved", len(records))

tab_check, tab_records, tab_data, tab_about = st.tabs(
    ["🩺 New check", "📋 Patient records", "📊 Dataset explorer", "ℹ️ About"]
)


# ---------------------------------------------------------------- result renderer
def render_result(res):
    css = {"SAFE": "ok", "NOT RECOMMENDED": "bad", "REVIEW": "warn"}[res["verdict"]]
    icon = {"SAFE": "✅", "NOT RECOMMENDED": "❌", "REVIEW": "⚠️"}[res["verdict"]]
    text = {"SAFE": "appears SAFE", "NOT RECOMMENDED": "is NOT recommended",
            "REVIEW": "needs manual review"}[res["verdict"]]
    who = f" for {res['patient_name']}" if res["patient_name"] else ""
    st.markdown(f'<div class="result {css}">{icon} {res["drug"]} {text}{who}</div>', unsafe_allow_html=True)

    p = res["prob"]
    color = "#22c55e" if p >= 0.7 else "#f59e0b" if p >= 0.4 else "#ef4444"
    st.markdown(f"**AI confidence the medicine is safe: {p * 100:.1f}%**")
    st.markdown(f'<div class="bar"><div style="width:{p * 100:.0f}%;background:{color}"></div></div>',
                unsafe_allow_html=True)

    st.markdown("")
    if res["reasons"]:
        st.markdown("**Why:**")
        for r in res["reasons"]:
            st.markdown(f'<div class="reason">⚠️ {r}</div>', unsafe_allow_html=True)
    else:
        st.info("No allergy, interaction or condition conflicts found in the knowledge base.")

    chips = [f"Age {res['age']}", res["gender"]]
    if res["pregnant"]:
        chips.append("🤰 Pregnant")
    if res["kidney"]:
        chips.append("Kidney disease")
    if res["liver"]:
        chips.append("Liver disease")
    chips += [f"Allergy: {a}" for a in res["allergies"]] + [f"On: {m}" for m in res["prev"]]
    st.markdown("".join(f'<span class="chip">{c}</span>' for c in chips), unsafe_allow_html=True)
    if res.get("saved"):
        st.success(f"Saved to patient records (ID {res['record_id']})")


# ---------------------------------------------------------------- TAB 1: check
with tab_check:
    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("👤 Patient information")
            a, b = st.columns(2)
            a.text_input("Patient name", key="p_name", placeholder="e.g. Anita Sharma")
            b.text_input("Doctor name", key="p_doctor", placeholder="e.g. Dr. Mehta")
            c, d = st.columns(2)
            c.number_input("Age (years)", min_value=0, max_value=110, key="p_age")
            d.radio("Gender", ["Male", "Female"], horizontal=True, key="p_gender")
            if st.session_state["p_gender"] == "Female":
                st.toggle("🤰 Patient is pregnant", key="p_pregnant")
            else:
                st.caption("🤰 The pregnancy option appears when **Female** is selected.")
            e, f = st.columns(2)
            e.toggle("Kidney disease", key="p_kidney")
            f.toggle("Liver disease", key="p_liver")

        with st.container(border=True):
            st.subheader("🧬 Medical history")
            st.multiselect("Known allergies (drug class)", ALLERGY_OPTIONS, key="p_allergies",
                           format_func=lambda x: x.replace("_", " ").title())
            st.multiselect("Previous / current medicines", sorted(DRUGS), key="p_prev")

    with right:
        with st.container(border=True):
            st.subheader("💊 Doctor's prescription")
            st.selectbox("Prescribed medicine", sorted(DRUGS), key="p_rx")
            st.caption(f"Drug class: **{DRUGS[st.session_state['p_rx']]['cls'].replace('_', ' ')}**")
            st.text_area("Diagnosis / notes (optional)", key="p_notes", height=80)
            save_it = st.checkbox("Save this patient to records", value=True)
            go = st.button("🔍 Check safety", type="primary", width="stretch")

        if go:
            ss = st.session_state
            female = ss["p_gender"] == "Female"
            patient = dict(age=int(ss["p_age"]), gender=ss["p_gender"],
                           pregnant=int(female and ss["p_pregnant"]),
                           kidney_disease=int(ss["p_kidney"]), liver_disease=int(ss["p_liver"]),
                           allergies=list(ss["p_allergies"]), current_meds=list(ss["p_prev"]))
            drug = ss["p_rx"]
            feats = pd.DataFrame([build_features(patient, drug)])[columns]
            prob = float(model.predict_proba(feats)[0][1])
            reasons = check_safety(patient, drug)
            verdict = "NOT RECOMMENDED" if reasons else ("SAFE" if prob >= 0.5 else "REVIEW")
            res = dict(drug=drug, prob=prob, reasons=reasons, verdict=verdict,
                       patient_name=ss["p_name"].strip(), age=patient["age"], gender=patient["gender"],
                       pregnant=patient["pregnant"], kidney=patient["kidney_disease"],
                       liver=patient["liver_disease"], allergies=patient["allergies"],
                       prev=patient["current_meds"], saved=False)
            if save_it:
                rid = uuid.uuid4().hex[:8].upper()
                save_record(dict(
                    record_id=rid, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    patient_name=ss["p_name"].strip() or "Unknown", doctor_name=ss["p_doctor"].strip(),
                    age=patient["age"], gender=patient["gender"], pregnant=patient["pregnant"],
                    kidney_disease=patient["kidney_disease"], liver_disease=patient["liver_disease"],
                    allergies=";".join(patient["allergies"]), previous_medicines=";".join(patient["current_meds"]),
                    prescribed_medicine=drug, diagnosis_notes=ss["p_notes"].strip(), verdict=verdict,
                    ml_prob_safe=round(prob, 4), reasons=" | ".join(reasons) if reasons else "No issues found"))
                res.update(saved=True, record_id=rid)
            st.session_state["result"] = res

        if "result" in st.session_state:
            render_result(st.session_state["result"])
        else:
            st.info("Fill in the patient details and press **Check safety**.")

# ---------------------------------------------------------------- TAB 2: records
with tab_records:
    st.subheader("📋 Saved patient records")
    recs = load_records()
    if recs.empty:
        st.info("No records yet. Run a check with “Save this patient to records” turned on.")
    else:
        q = st.text_input("🔎 Search (name, doctor, medicine, verdict...)", key="rec_search")
        view = recs.iloc[::-1].reset_index(drop=True)
        if q:
            mask = view.astype(str).apply(lambda col: col.str.contains(q, case=False, na=False)).any(axis=1)
            view = view[mask]
        s1, s2, s3 = st.columns(3)
        s1.metric("Records shown", len(view))
        s2.metric("Safe", int((view["verdict"] == "SAFE").sum()))
        s3.metric("Not recommended", int((view["verdict"] == "NOT RECOMMENDED").sum()))
        st.dataframe(view, width="stretch", height=420, hide_index=True)
        st.download_button("⬇️ Download records (CSV)", recs.to_csv(index=False).encode(),
                           "patient_records.csv", "text/csv")
        with st.expander("🗑️ Delete all records"):
            if st.checkbox("I understand this cannot be undone"):
                if st.button("Delete everything", type="primary"):
                    pd.DataFrame(columns=RECORD_COLS).to_csv(RECORDS_FILE, index=False)
                    st.rerun()
    st.caption("Note: on Streamlit Community Cloud the saved file is cleared when the app restarts, "
               "so download the CSV regularly (or connect a database for permanent storage).")

# ---------------------------------------------------------------- TAB 3: dataset
with tab_data:
    st.subheader("📊 Full dataset explorer")
    f1, f2, f3 = st.columns([2, 1, 2])
    meds = f1.multiselect("Filter by prescribed medicine", sorted(DRUGS))
    outcome = f2.radio("Outcome", ["All", "Safe", "Unsafe"], horizontal=True)
    text = f3.text_input("Search (allergy, reason, patient ID...)")

    view = data
    if meds:
        view = view[view["prescribed_medicine"].isin(meds)]
    if outcome != "All":
        view = view[view["is_safe"] == (1 if outcome == "Safe" else 0)]
    if text:
        mask = view.astype(str).apply(lambda col: col.str.contains(text, case=False, na=False)).any(axis=1)
        view = view[mask]

    st.caption(f"Showing **{len(view):,}** of **{len(data):,}** patients (scroll the table to see all rows)")
    st.dataframe(view, width="stretch", height=480, hide_index=True)
    st.download_button("⬇️ Download shown rows (CSV)", view.to_csv(index=False).encode(),
                       "medicine_dataset_filtered.csv", "text/csv")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Safe prescriptions (%) by medicine**")
        st.bar_chart((data.groupby("prescribed_medicine")["is_safe"].mean() * 100).sort_values())
    with g2:
        st.markdown("**Safe vs unsafe count**")
        st.bar_chart(data["is_safe"].map({1: "Safe", 0: "Unsafe"}).value_counts())

# ---------------------------------------------------------------- TAB 4: about
with tab_about:
    st.markdown(
        """
### How it works
1. Patient details, allergies and medicine history are turned into features.
2. A **Random Forest** model trained on the dataset predicts the chance that the prescription is safe.
3. A **rule engine** checks allergies (including cross-reactions), drug interactions, pregnancy,
   kidney/liver disease and minimum age, and explains the result.
4. Every check can be saved to **Patient records** for later review.

### Verdicts
- ✅ **Safe**: no rule conflicts and the model agrees.
- ❌ **Not recommended**: at least one rule conflict was found.
- ⚠️ **Review**: no rule conflict but the model is unsure.

**Disclaimer:** the dataset is synthetic and the rules are simplified. This is a learning project,
not a medical device.
"""
    )
