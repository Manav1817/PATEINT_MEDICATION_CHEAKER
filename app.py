import os
import uuid
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

from knowledge import (DRUGS, CONDITIONS, ALLERGY_LABELS, check_safety_detailed,
                       build_features, row_to_patient)

st.set_page_config(
    page_title="MediSafe | AI Clinical Safety Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

RECORDS_FILE = "patient_records.csv"
DATASET_FILE = "medicine_dataset.csv"
RECORD_COLS = ["record_id", "timestamp", "patient_name", "doctor_name", "age", "gender",
               "conditions", "allergies", "previous_medicines", "prescribed_medicine",
               "notes", "verdict", "ml_prob_safe", "reasons"]

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, button, input, textarea, select {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Hide default streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
}
header[data-testid="stHeader"] {
    background: transparent;
    height: 0;
}

/* Main Container Adjustments */
.block-container {
    max-width: 1240px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* Top App Header Banner */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 20px;
    padding: 1.25rem 1.75rem;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
    margin-bottom: 1.5rem;
}
.brand-box {
    display: flex;
    align-items: center;
    gap: 1rem;
}
.brand-logo {
    background: linear-gradient(135deg, #0284c7, #0d9488);
    color: #ffffff;
    border-radius: 14px;
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.3);
}
.brand-title {
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #ffffff;
    margin: 0;
}
.brand-sub {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: -2px;
}
.header-badge {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    padding: 0.4rem 0.9rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #38bdf8;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* Tab Bar Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 2px;
}
.stTabs [data-baseweb="tab"] {
    height: 48px;
    padding: 0 1.5rem;
    font-weight: 600;
    font-size: 0.95rem;
    border-radius: 10px 10px 0 0;
    color: #64748b;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #0284c7 !important;
    background-color: #f0f9ff !important;
    border-bottom: 3px solid #0284c7 !important;
}

/* Section Header Titles */
.section-head {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 1rem;
}

/* Verdict Cards */
.verdict-card {
    border-radius: 18px;
    padding: 1.25rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.25rem;
    margin-top: 0.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.05);
}
.verdict-card.ok {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 1px solid #86efac;
}
.verdict-card.bad {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border: 1px solid #fca5a5;
}
.verdict-card.warn {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border: 1px solid #fcd34d;
}
.verdict-card h3 {
    margin: 0;
    font-size: 1.35rem;
    font-weight: 700;
    color: #0f172a;
}
.verdict-card p {
    margin: 0.2rem 0 0 0;
    color: #475569;
    font-size: 0.92rem;
}

/* Safety Gauge Indicator */
.gauge-ring {
    width: 76px;
    height: 76px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 4px 10px rgba(0,0,0,0.06);
}
.gauge-center {
    width: 58px;
    height: 58px;
    border-radius: 50%;
    background: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 1.1rem;
    color: #0f172a;
}

/* Check list item */
.check-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.65rem 0.5rem;
    border-bottom: 1px dashed #e2e8f0;
    font-size: 0.92rem;
    font-weight: 500;
    color: #334155;
}
.check-pass {
    color: #16a34a;
    background: #dcfce7;
    padding: 0.2rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 700;
}
.check-fail {
    color: #dc2626;
    background: #fee2e2;
    padding: 0.2rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 700;
}

/* Issue / Warning Box */
.issue-box {
    padding: 0.75rem 1rem;
    border-left: 4px solid #ef4444;
    background: #fef2f2;
    border-radius: 8px;
    margin: 0.5rem 0;
    color: #991b1b;
    font-size: 0.9rem;
    font-weight: 500;
}

/* Safer Alternatives Chips */
.alt-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.45rem 0.9rem;
    border-radius: 9999px;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    margin: 0.3rem 0.4rem 0.3rem 0;
    font-size: 0.88rem;
    font-weight: 600;
    color: #1e293b;
}
.alt-score {
    color: #059669;
    background: #dcfce7;
    padding: 0.1rem 0.45rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
}

/* Stats Cards */
.stat-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
}
.stat-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #0f172a;
}
.stat-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Footer Disclaimer */
.footer-note {
    text-align: center;
    color: #94a3b8;
    font-size: 0.82rem;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid #e2e8f0;
}
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
    css, icon, title = {
        "SAFE": ("ok", "✅", "Safe to Prescribe"),
        "NOT RECOMMENDED": ("bad", "⛔", "Not Recommended"),
        "REVIEW": ("warn", "⚠️", "Requires Medical Review")
    }[r["verdict"]]
    color = {"ok": "#10b981", "bad": "#ef4444", "warn": "#f59e0b"}[css]
    pct = round(r["prob"] * 100)
    who = r["name"] or "Selected Patient"
    
    st.markdown(
        f'''
        <div class="verdict-card {css}">
            <div class="gauge-ring" style="background: conic-gradient({color} {pct}%, #e2e8f0 0);">
                <div class="gauge-center">{pct}%</div>
            </div>
            <div>
                <h3>{icon} {title}</h3>
                <p>Prescribing <b>{r["drug"]}</b> for <b>{who}</b></p>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    bad_cats = {c for c, _ in r["issues"]}
    rows = ""
    for cat in ["Allergies", "Drug interactions", "Medical conditions", "Age"]:
        status = '<span class="check-fail">✕ Conflict</span>' if cat in bad_cats else '<span class="check-pass">✓ Clear</span>'
        rows += f'<div class="check-item"><span>{cat}</span>{status}</div>'
    
    st.markdown(f'<div style="margin-bottom: 1rem;">{rows}</div>', unsafe_allow_html=True)

    for _, msg in r["issues"]:
        st.markdown(f'<div class="issue-box">⚠️ <b>Warning:</b> {msg}</div>', unsafe_allow_html=True)

    if r["alternatives"]:
        st.markdown('<div class="section-head" style="margin-top:1.2rem;">✨ Safer Recommended Alternatives</div>', unsafe_allow_html=True)
        chips = "".join([f'<span class="alt-chip">{n} <span class="alt-score">{round(p * 100)}% Match</span></span>'
                         for n, p in r["alternatives"]])
        st.markdown(chips, unsafe_allow_html=True)
    elif r["verdict"] != "SAFE":
        st.caption("ℹ️ No safer alternative found in the same drug category.")
        
    if r.get("saved"):
        st.success(f"✓ Saved to records (Record ID: **{r['record_id']}**)")


# ---------------------------------------------------------------- top header
st.markdown(
    """
    <div class="app-header">
        <div class="brand-box">
            <div class="brand-logo">💊</div>
            <div>
                <div class="brand-title">MediSafe AI</div>
                <div class="brand-sub">Clinical Prescription Safety & Interaction Analyzer</div>
            </div>
        </div>
        <div class="header-badge">
            <span style="display:inline-block; width:8px; height:8px; background:#38bdf8; border-radius:50%;"></span>
            RandomForest Model v2.4 Active
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

tab_check, tab_rec, tab_data = st.tabs(["🔬 Prescription Safety Check", "📋 Patient Records", "📊 Training Dataset"])

# ---------------------------------------------------------------- tab: check
with tab_check:
    left, right = st.columns([5, 4], gap="large")

    with left:
        with st.container(border=True):
            h, pop = st.columns([3, 2])
            h.markdown('<div class="section-head">👤 Patient Details</div>', unsafe_allow_html=True)
            with pop.popover("⚡ Preset Patients"):
                for label in EXAMPLES:
                    st.button(label, key=f"ex_{label}", on_click=load_example, args=(label,), use_container_width=True)

            a, b = st.columns(2)
            a.text_input("Patient Name", key="p_name", placeholder="e.g. John Doe")
            b.text_input("Attending Doctor (Optional)", key="p_doctor", placeholder="e.g. Dr. Smith")
            
            c, d = st.columns(2)
            c.number_input("Age (Years)", min_value=0, max_value=110, key="p_age")
            d.radio("Gender", ["Male", "Female"], horizontal=True, key="p_gender")
            
            st.multiselect("Medical Conditions", list(CONDITIONS), key="p_conditions",
                           format_func=lambda x: CONDITIONS[x], placeholder="Select conditions (e.g., Kidney disease, Diabetes)")
            
            if "pregnancy" in st.session_state["p_conditions"] and st.session_state["p_gender"] != "Female":
                st.warning("Note: Pregnancy condition applies only to female patients and will be omitted.")
                
            st.multiselect("Known Drug Allergies", list(ALLERGY_LABELS), key="p_allergies",
                           format_func=lambda x: ALLERGY_LABELS[x], placeholder="Select known allergies")
            
            st.multiselect("Current Active Medications", sorted(DRUGS), key="p_prev",
                           placeholder="Select currently prescribed drugs")

    with right:
        with st.container(border=True):
            st.markdown('<div class="section-head">💊 Proposed Prescription</div>', unsafe_allow_html=True)
            st.selectbox("Prescribed Medicine", sorted(DRUGS), key="p_rx",
                         format_func=lambda n: f"{n}  —  ({DRUGS[n]['group']})")
            
            with st.expander("📝 Additional Clinical Notes"):
                st.text_area("Diagnosis / Instructions", key="p_notes", height=80, label_visibility="collapsed")
            
            save_it = st.checkbox("Save analysis to Patient Records", value=True)
            
            b1, b2 = st.columns([3, 1])
            go = b1.button("Check Prescription Safety", type="primary", use_container_width=True)
            b2.button("Reset", on_click=clear_form, use_container_width=True)

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
                    patient_name=ss["p_name"].strip() or "Anonymous", doctor_name=ss["p_doctor"].strip() or "N/A",
                    age=patient["age"], gender=patient["gender"], conditions=";".join(patient["conditions"]),
                    allergies=";".join(patient["allergies"]), previous_medicines=";".join(patient["current_meds"]),
                    prescribed_medicine=drug, notes=ss["p_notes"].strip(), verdict=verdict,
                    ml_prob_safe=round(prob, 4),
                    reasons=" | ".join(m for _, m in issues) if issues else "No safety issues identified"))
                res.update(saved=True, record_id=rid)
            ss["result"] = res

        if "result" in st.session_state:
            render_result(st.session_state["result"])

# ---------------------------------------------------------------- tab: records
with tab_rec:
    recs = load_records()
    
    # Record Metrics
    if not recs.empty:
        total = len(recs)
        safe_cnt = len(recs[recs["verdict"] == "SAFE"])
        flagged_cnt = total - safe_cnt
        
        m1, m2, m3 = st.columns(3)
        m1.markdown(f'<div class="stat-card"><div class="stat-value">{total}</div><div class="stat-label">Total Checks</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#10b981;">{safe_cnt}</div><div class="stat-label">Passed Safe</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#ef4444;">{flagged_cnt}</div><div class="stat-label">Flagged / Conflicts</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    if recs.empty:
        st.info("No patient records saved yet. Perform a safety check with 'Save analysis' enabled.")
    else:
        q = st.text_input("🔍 Search Patient History", placeholder="Search by patient, doctor, medicine, or diagnosis...",
                          label_visibility="collapsed", key="rec_q")
        view = pd.DataFrame({
            "ID": recs["record_id"], "Date": recs["timestamp"], "Patient": recs["patient_name"],
            "Doctor": recs["doctor_name"], "Age": recs["age"], "Gender": recs["gender"],
            "Conditions": recs["conditions"].apply(lambda s: pretty(s, CONDITIONS)),
            "Allergies": recs["allergies"].apply(lambda s: pretty(s, ALLERGY_LABELS)),
            "Current Medicines": recs["previous_medicines"].apply(pretty),
            "Prescribed": recs["prescribed_medicine"], "Verdict": recs["verdict"],
            "Safety Score": (recs["ml_prob_safe"].astype(float) * 100).round(0),
            "Details / Reasons": recs["reasons"], "Notes": recs["notes"],
        }).iloc[::-1]
        
        if q:
            mask = view.astype(str).apply(lambda col: col.str.contains(q, case=False, na=False)).any(axis=1)
            view = view[mask]
            
        st.caption(f"Showing {len(view)} patient record(s)")
        st.dataframe(
            view,
            use_container_width=True,
            height=400,
            hide_index=True,
            column_config={
                "Safety Score": st.column_config.ProgressColumn(
                    "Safety Score", min_value=0, max_value=100, format="%d%%"
                )
            }
        )
        
        d1, d2 = st.columns([1, 1])
        d1.download_button("📥 Export Records (CSV)", recs.to_csv(index=False).encode(), "patient_records.csv", "text/csv")
        with d2.popover("🗑️ Clear All Records"):
            st.write("Are you sure you want to permanently clear all records?")
            if st.button("Yes, delete all records", type="primary"):
                pd.DataFrame(columns=RECORD_COLS).to_csv(RECORDS_FILE, index=False)
                st.rerun()

# ---------------------------------------------------------------- tab: dataset
with tab_data:
    st.markdown('<div class="section-head">🔍 Reference Safety Dataset</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns([3, 2, 3])
    meds = f1.multiselect("Filter Medicine", sorted(DRUGS), placeholder="Filter by prescribed medicine",
                          label_visibility="collapsed")
    outcome = f2.radio("Outcome Filter", ["All", "Safe Only", "Unsafe Only"], horizontal=True, label_visibility="collapsed")
    text = f3.text_input("Filter Text", placeholder="Search conditions, allergies, or reasons...",
                         label_visibility="collapsed", key="data_q")
    
    view = data
    if meds:
        view = view[view["prescribed_medicine"].isin(meds)]
    if outcome == "Safe Only":
        view = view[view["is_safe"] == 1]
    elif outcome == "Unsafe Only":
        view = view[view["is_safe"] == 0]
        
    if text:
        mask = view.astype(str).apply(lambda col: col.str.contains(text, case=False, na=False)).any(axis=1)
        view = view[mask]
        
    st.caption(f"Displaying {len(view):,} of {len(data):,} entries")
    st.dataframe(view, use_container_width=True, height=480, hide_index=True)
    st.download_button("📥 Download Dataset CSV", view.to_csv(index=False).encode(), "medicine_dataset.csv", "text/csv")

st.markdown('<div class="footer-note">⚡ MediSafe AI Clinical Assistant · Educational & Decision Support Use Only · Always Consult Clinical Guidelines</div>',
            unsafe_allow_html=True)