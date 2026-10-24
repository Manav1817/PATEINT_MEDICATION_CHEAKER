import os
import uuid
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

from knowledge import (DRUGS, CONDITIONS, ALLERGY_LABELS, check_safety_detailed,
                       build_features, row_to_patient)

st.set_page_config(page_title="MediSafe", page_icon="💊", layout="wide",
                   initial_sidebar_state="collapsed")

RECORDS_FILE = "patient_records.csv"
DATASET_FILE = "medicine_dataset.csv"
RECORD_COLS = ["record_id", "timestamp", "patient_name", "doctor_name", "age", "gender",
               "conditions", "allergies", "previous_medicines", "prescribed_medicine",
               "notes", "verdict", "ml_prob_safe", "reasons"]

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea {font-family:'Inter',sans-serif;}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="collapsedControl"] {display:none !important;}
header[data-testid="stHeader"] {background:transparent; height:0;}
.block-container {max-width:1180px; padding-top:1.3rem; padding-bottom:2rem;}
.brand {display:flex; align-items:center; gap:.65rem; font-size:1.55rem; font-weight:700; color:#0f172a; margin-bottom:.5rem;}
.brand .logo {background:linear-gradient(135deg,#0d9488,#2563eb); color:#fff; border-radius:12px;
              width:42px; height:42px; display:flex; align-items:center; justify-content:center; font-size:1.3rem;}
.stTabs [data-baseweb="tab-list"] {gap:.4rem; border-bottom:1px solid #e2e8f0;}
.stTabs [data-baseweb="tab"] {height:46px; padding:0 1.1rem; font-weight:600;}
.title {font-weight:600; font-size:1.05rem; color:#0f172a; margin:0 0 .4rem 0;}
.verdict {border-radius:16px; padding:1.1rem 1.3rem; display:flex; gap:1.1rem; align-items:center; margin-top:.4rem;}
.verdict.ok {background:#ecfdf5; border:1px solid #6ee7b7;}
.verdict.bad {background:#fef2f2; border:1px solid #fca5a5;}
.verdict.warn {background:#fffbeb; border:1px solid #fcd34d;}
.verdict h3 {margin:0; font-size:1.3rem; color:#0f172a; padding:0;}
.verdict small {color:#475569; font-size:.9rem;}
.gauge {width:76px; height:76px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex-shrink:0;}
.gauge span {width:58px; height:58px; border-radius:50%; background:#fff; display:flex; align-items:center;
             justify-content:center; font-weight:700; color:#0f172a;}
.check {display:flex; justify-content:space-between; padding:.55rem .2rem; border-bottom:1px solid #f1f5f9; font-size:.95rem; color:#0f172a;}
.pass {color:#059669; font-weight:600;}
.fail {color:#dc2626; font-weight:600;}
.reason {padding:.5rem .8rem; border-left:4px solid #ef4444; background:#fef2f2; border-radius:6px; margin:.4rem 0; color:#7f1d1d; font-size:.92rem;}
.alt {display:inline-block; padding:.35rem .85rem; border-radius:999px; background:#f1f5f9; margin:.2rem .3rem .2rem 0; font-weight:500; color:#0f172a;}
.alt b {color:#059669;}
.foot {text-align:center; color:#94a3b8; font-size:.8rem; margin-top:2.2rem;}
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
        # Fallback: retrain if model.pkl is missing or was built with another scikit-learn version
        d = pd.read_csv(DATASET_FILE)
        X = pd.DataFrame([build_features(row_to_patient(r), r["prescribed_medicine"]) for _, r in d.iterrows()])
        m = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced",
                                   random_state=42, n_jobs=-1).fit(X, d["is_safe"])
        return m, list(X.columns)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_FILE)


def load_records():
    try:
        if os.path.exists(RECORDS_FILE) and os.path.getsize(RECORDS_FILE) > 0:
            return pd.read_csv(RECORDS_FILE).reindex(columns=RECORD_COLS)
    except Exception:
        pass
    return pd.DataFrame(columns=RECORD_COLS)


def save_record(rec):
    pd.concat([load_records(), pd.DataFrame([rec])], ignore_index=True).to_csv(RECORDS_FILE, index=False)


def pretty(s, mapping=None):
    items = [x for x in str(s).split(";") if x and x != "nan"]
    return ", ".join(mapping.get(x, x) for x in items) if mapping else ", ".join(items)


model, columns = load_model()
data = load_dataset()

# ---------------------------------------------------------------- session state
DEFAULTS = dict(p_name="", p_doctor="", p_age=35, p_gender="Male", p_conditions=[],
                p_allergies=[], p_prev=[], p_rx="Amoxicillin", p_notes="")
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

EXAMPLES = {
    "Penicillin allergy": dict(p_name="Anita Sharma", p_age=28, p_gender="Female", p_conditions=[],
                               p_allergies=["penicillin"], p_prev=["Cetirizine"], p_rx="Amoxicillin"),
    "On a blood thinner": dict(p_name="Ravi Patil", p_age=58, p_gender="Male", p_conditions=["hypertension"],
                               p_allergies=[], p_prev=["Warfarin", "Atorvastatin"], p_rx="Ibuprofen"),
    "Pregnant patient": dict(p_name="Sneha Kulkarni", p_age=26, p_gender="Female", p_conditions=["pregnancy"],
                             p_allergies=[], p_prev=[], p_rx="Doxycycline"),
    "Asthma patient": dict(p_name="Imran Shaikh", p_age=40, p_gender="Male", p_conditions=["asthma", "diabetes"],
                           p_allergies=[], p_prev=["Metformin"], p_rx="Propranolol"),
}


def load_example(key):
    st.session_state.update(DEFAULTS)
    st.session_state.update(EXAMPLES[key])
    st.session_state.pop("result", None)


def clear_form():
    st.session_state.update(DEFAULTS)
    st.session_state.pop("result", None)


# ---------------------------------------------------------------- helpers
def build_patient():
    ss = st.session_state
    conds = [c for c in ss["p_conditions"] if not (c == "pregnancy" and ss["p_gender"] != "Female")]
    return dict(age=int(ss["p_age"]), gender=ss["p_gender"], conditions=conds,
                allergies=list(ss["p_allergies"]), current_meds=list(ss["p_prev"]))


def find_alternatives(patient, drug):
    group = DRUGS[drug]["group"]
    ok = [n for n, d in DRUGS.items()
          if d["group"] == group and n != drug and not check_safety_detailed(patient, n)]
    if not ok:
        return []
    X = pd.DataFrame([build_features(patient, n) for n in ok])[columns]
    probs = model.predict_proba(X)[:, 1]
    ranked = sorted(zip(ok, probs), key=lambda t: -t[1])
    return [(n, float(p)) for n, p in ranked if p >= 0.5][:5]


def render_result(r):
    css, icon, title = {"SAFE": ("ok", "✅", "Safe to prescribe"),
                        "NOT RECOMMENDED": ("bad", "⛔", "Not recommended"),
                        "REVIEW": ("warn", "⚠️", "Needs review")}[r["verdict"]]
    color = {"ok": "#10b981", "bad": "#ef4444", "warn": "#f59e0b"}[css]
    pct = round(r["prob"] * 100)
    who = r["name"] or "this patient"
    st.markdown(
        f'<div class="verdict {css}">'
        f'<div class="gauge" style="background:conic-gradient({color} {pct}%, #e2e8f0 0)"><span>{pct}%</span></div>'
        f'<div><h3>{icon} {title}</h3><small>{r["drug"]} for {who}</small></div></div>',
        unsafe_allow_html=True,
    )

    bad_cats = {c for c, _ in r["issues"]}
    rows = ""
    for cat in ["Allergies", "Drug interactions", "Medical conditions", "Age"]:
        status = '<span class="fail">✕ Issue found</span>' if cat in bad_cats else '<span class="pass">✓ Clear</span>'
        rows += f'<div class="check"><span>{cat}</span>{status}</div>'
    st.markdown(f'<div style="margin-top:.8rem">{rows}</div>', unsafe_allow_html=True)

    for _, msg in r["issues"]:
        st.markdown(f'<div class="reason">{msg}</div>', unsafe_allow_html=True)

    if r["alternatives"]:
        st.markdown('<div class="title" style="margin-top:1rem">Safer alternatives</div>', unsafe_allow_html=True)
        st.markdown("".join(f'<span class="alt">{n} · <b>{round(p * 100)}%</b></span>'
                            for n, p in r["alternatives"]), unsafe_allow_html=True)
    elif r["verdict"] != "SAFE":
        st.caption("No safer alternative found in the same medicine group.")
    if r.get("saved"):
        st.caption(f"Saved to Records · ID {r['record_id']}")


# ---------------------------------------------------------------- header
st.markdown('<div class="brand"><div class="logo">💊</div>MediSafe</div>', unsafe_allow_html=True)
tab_check, tab_rec, tab_data = st.tabs(["Check prescription", "Records", "Dataset"])

# ---------------------------------------------------------------- tab: check
with tab_check:
    left, right = st.columns([5, 4], gap="large")

    with left:
        with st.container(border=True):
            h, pop = st.columns([3, 2])
            h.markdown('<div class="title">Patient</div>', unsafe_allow_html=True)
            with pop.popover("Sample patients"):
                for label in EXAMPLES:
                    st.button(label, key=f"ex_{label}", on_click=load_example, args=(label,), width="stretch")

            a, b = st.columns(2)
            a.text_input("Patient name", key="p_name", placeholder="Full name")
            b.text_input("Doctor (optional)", key="p_doctor", placeholder="Dr. name")
            c, d = st.columns(2)
            c.number_input("Age (years)", min_value=0, max_value=110, key="p_age")
            d.radio("Gender", ["Male", "Female"], horizontal=True, key="p_gender")
            st.multiselect("Medical conditions", list(CONDITIONS), key="p_conditions",
                           format_func=lambda x: CONDITIONS[x], placeholder="Select conditions (pregnancy, diabetes, asthma...)")
            if "pregnancy" in st.session_state["p_conditions"] and st.session_state["p_gender"] != "Female":
                st.warning("Pregnancy applies to female patients only and will be ignored.")
            st.multiselect("Drug allergies", list(ALLERGY_LABELS), key="p_allergies",
                           format_func=lambda x: ALLERGY_LABELS[x], placeholder="Select allergies")
            st.multiselect("Previous / current medicines", sorted(DRUGS), key="p_prev",
                           placeholder="Select medicines the patient takes")

    with right:
        with st.container(border=True):
            st.markdown('<div class="title">Prescription</div>', unsafe_allow_html=True)
            st.selectbox("Prescribed medicine", sorted(DRUGS), key="p_rx",
                         format_func=lambda n: f"{n}  ·  {DRUGS[n]['group']}")
            with st.expander("Add notes"):
                st.text_area("Diagnosis / notes", key="p_notes", height=80, label_visibility="collapsed")
            save_it = st.checkbox("Save to records", value=True)
            b1, b2 = st.columns([3, 1])
            go = b1.button("Check safety", type="primary", width="stretch")
            b2.button("Clear", on_click=clear_form, width="stretch")

        if go:
            ss = st.session_state
            patient = build_patient()
            drug = ss["p_rx"]
            issues = check_safety_detailed(patient, drug)
            prob = float(model.predict_proba(pd.DataFrame([build_features(patient, drug)])[columns])[0][1])
            verdict = "NOT RECOMMENDED" if issues else ("SAFE" if prob >= 0.5 else "REVIEW")
            res = dict(drug=drug, prob=prob, issues=issues, verdict=verdict, name=ss["p_name"].strip(),
                       alternatives=find_alternatives(patient, drug) if verdict != "SAFE" else [], saved=False)
            if save_it:
                rid = uuid.uuid4().hex[:8].upper()
                save_record(dict(
                    record_id=rid, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    patient_name=ss["p_name"].strip() or "Unknown", doctor_name=ss["p_doctor"].strip(),
                    age=patient["age"], gender=patient["gender"], conditions=";".join(patient["conditions"]),
                    allergies=";".join(patient["allergies"]), previous_medicines=";".join(patient["current_meds"]),
                    prescribed_medicine=drug, notes=ss["p_notes"].strip(), verdict=verdict,
                    ml_prob_safe=round(prob, 4),
                    reasons=" | ".join(m for _, m in issues) if issues else "No issues found"))
                res.update(saved=True, record_id=rid)
            ss["result"] = res

        if "result" in st.session_state:
            render_result(st.session_state["result"])

# ---------------------------------------------------------------- tab: records
with tab_rec:
    recs = load_records()
    if recs.empty:
        st.info("No records yet. Run a check with “Save to records” turned on.")
    else:
        q = st.text_input("Search", placeholder="Search by patient, doctor, medicine or result",
                          label_visibility="collapsed", key="rec_q")
        view = pd.DataFrame({
            "ID": recs["record_id"], "Date": recs["timestamp"], "Patient": recs["patient_name"],
            "Doctor": recs["doctor_name"], "Age": recs["age"], "Gender": recs["gender"],
            "Conditions": recs["conditions"].apply(lambda s: pretty(s, CONDITIONS)),
            "Allergies": recs["allergies"].apply(lambda s: pretty(s, ALLERGY_LABELS)),
            "Current medicines": recs["previous_medicines"].apply(pretty),
            "Prescribed": recs["prescribed_medicine"], "Result": recs["verdict"],
            "Safe score": (recs["ml_prob_safe"].astype(float) * 100).round(0),
            "Details": recs["reasons"], "Notes": recs["notes"],
        }).iloc[::-1]
        if q:
            mask = view.astype(str).apply(lambda col: col.str.contains(q, case=False, na=False)).any(axis=1)
            view = view[mask]
        st.caption(f"{len(view)} record(s)")
        st.dataframe(view, width="stretch", height=430, hide_index=True,
                     column_config={"Safe score": st.column_config.ProgressColumn(
                         "Safe score", min_value=0, max_value=100, format="%d%%")})
        d1, d2 = st.columns([1, 1])
        d1.download_button("Download CSV", recs.to_csv(index=False).encode(), "patient_records.csv", "text/csv")
        with d2.popover("Delete all records"):
            st.write("This cannot be undone.")
            if st.button("Yes, delete everything", type="primary"):
                pd.DataFrame(columns=RECORD_COLS).to_csv(RECORDS_FILE, index=False)
                st.rerun()

# ---------------------------------------------------------------- tab: dataset
with tab_data:
    f1, f2, f3 = st.columns([3, 2, 3])
    meds = f1.multiselect("Medicine", sorted(DRUGS), placeholder="Filter by prescribed medicine",
                          label_visibility="collapsed")
    outcome = f2.radio("Outcome", ["All", "Safe", "Unsafe"], horizontal=True, label_visibility="collapsed")
    text = f3.text_input("Search", placeholder="Search patients, allergies, reasons...",
                         label_visibility="collapsed", key="data_q")
    view = data
    if meds:
        view = view[view["prescribed_medicine"].isin(meds)]
    if outcome != "All":
        view = view[view["is_safe"] == (1 if outcome == "Safe" else 0)]
    if text:
        mask = view.astype(str).apply(lambda col: col.str.contains(text, case=False, na=False)).any(axis=1)
        view = view[mask]
    st.caption(f"{len(view):,} of {len(data):,} patients")
    st.dataframe(view, width="stretch", height=500, hide_index=True)
    st.download_button("Download CSV", view.to_csv(index=False).encode(), "medicine_dataset.csv", "text/csv")

st.markdown('<div class="foot">Educational use only · Not a substitute for professional medical advice</div>',
            unsafe_allow_html=True)
