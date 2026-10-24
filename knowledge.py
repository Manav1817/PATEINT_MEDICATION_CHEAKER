import random
import pandas as pd

# ------------------------------------------------------------------ medical conditions
CONDITIONS = {
    "pregnancy": "Pregnancy",
    "kidney_disease": "Kidney disease",
    "liver_disease": "Liver disease",
    "diabetes": "Diabetes",
    "hypertension": "High blood pressure",
    "heart_failure": "Heart failure",
    "asthma": "Asthma",
    "peptic_ulcer": "Stomach ulcer",
    "epilepsy": "Epilepsy",
    "glaucoma": "Glaucoma",
    "gout": "Gout",
    "bleeding_disorder": "Bleeding disorder",
    "g6pd_deficiency": "G6PD deficiency",
    "hyperthyroidism": "Overactive thyroid",
    "enlarged_prostate": "Enlarged prostate",
}

# ------------------------------------------------------------------ drug allergies
ALLERGY_LABELS = {
    "penicillin": "Penicillin",
    "cephalosporin": "Cephalosporins",
    "sulfa": "Sulfa drugs",
    "nsaid": "NSAIDs (ibuprofen, diclofenac...)",
    "aspirin": "Aspirin",
    "opioid": "Opioids",
    "macrolide": "Macrolide antibiotics",
    "fluoroquinolone": "Fluoroquinolone antibiotics",
    "tetracycline": "Tetracycline antibiotics",
    "statin": "Statins",
    "ace_inhibitor": "ACE inhibitors",
    "anticonvulsant": "Anticonvulsants",
    "azole": "Antifungal azoles",
}
ALLERGY_OPTIONS = list(ALLERGY_LABELS)

# allergy -> drug classes blocked (includes cross-reactivity)
CROSS = {
    "penicillin": ["penicillin", "cephalosporin"],
    "aspirin": ["aspirin", "nsaid"],
    "nsaid": ["nsaid", "aspirin"],
}


# ------------------------------------------------------------------ medicines
def D(group, cls, min_age=0, avoid=()):
    return dict(group=group, cls=cls, min_age=min_age, avoid=list(avoid))


DRUGS = {
    # Antibiotics
    "Amoxicillin": D("Antibiotic", "penicillin"),
    "Ampicillin": D("Antibiotic", "penicillin"),
    "Cephalexin": D("Antibiotic", "cephalosporin"),
    "Azithromycin": D("Antibiotic", "macrolide", avoid=["liver_disease"]),
    "Clarithromycin": D("Antibiotic", "macrolide", avoid=["liver_disease", "pregnancy"]),
    "Ciprofloxacin": D("Antibiotic", "fluoroquinolone", 18, ["pregnancy", "epilepsy"]),
    "Levofloxacin": D("Antibiotic", "fluoroquinolone", 18, ["pregnancy", "epilepsy"]),
    "Doxycycline": D("Antibiotic", "tetracycline", 8, ["pregnancy"]),
    "Cotrimoxazole": D("Antibiotic", "sulfa", 0, ["pregnancy", "kidney_disease", "g6pd_deficiency"]),
    "Nitrofurantoin": D("Antibiotic", "nitrofurantoin", 1, ["kidney_disease", "g6pd_deficiency"]),
    "Metronidazole": D("Antibiotic", "nitroimidazole", 0, ["liver_disease"]),
    # Antifungal / antiviral
    "Fluconazole": D("Antifungal / antiviral", "azole", 0, ["pregnancy", "liver_disease"]),
    "Acyclovir": D("Antifungal / antiviral", "antiviral", 0, ["kidney_disease"]),
    # Pain and fever
    "Paracetamol": D("Pain & fever", "analgesic", 0, ["liver_disease"]),
    "Ibuprofen": D("Pain & fever", "nsaid", 1, ["pregnancy", "kidney_disease", "peptic_ulcer", "heart_failure", "bleeding_disorder"]),
    "Diclofenac": D("Pain & fever", "nsaid", 12, ["pregnancy", "kidney_disease", "liver_disease", "peptic_ulcer", "heart_failure", "bleeding_disorder"]),
    "Naproxen": D("Pain & fever", "nsaid", 12, ["pregnancy", "kidney_disease", "peptic_ulcer", "heart_failure", "bleeding_disorder"]),
    "Aspirin": D("Pain & fever", "aspirin", 16, ["pregnancy", "kidney_disease", "peptic_ulcer", "bleeding_disorder", "asthma"]),
    # Strong pain relief
    "Tramadol": D("Strong pain relief", "opioid", 12, ["pregnancy", "kidney_disease", "liver_disease", "epilepsy"]),
    "Codeine": D("Strong pain relief", "opioid", 12, ["pregnancy", "asthma"]),
    # Blood thinners
    "Warfarin": D("Blood thinner", "anticoagulant", 18, ["pregnancy", "liver_disease", "peptic_ulcer", "bleeding_disorder"]),
    "Clopidogrel": D("Blood thinner", "antiplatelet", 18, ["peptic_ulcer", "bleeding_disorder", "liver_disease"]),
    # Diabetes
    "Metformin": D("Diabetes", "antidiabetic", 10, ["kidney_disease"]),
    "Glimepiride": D("Diabetes", "antidiabetic", 18, ["pregnancy", "kidney_disease", "liver_disease"]),
    # Cholesterol
    "Atorvastatin": D("Cholesterol", "statin", 10, ["pregnancy", "liver_disease"]),
    # Blood pressure and heart
    "Lisinopril": D("Blood pressure & heart", "ace_inhibitor", 6, ["pregnancy", "kidney_disease"]),
    "Losartan": D("Blood pressure & heart", "arb", 6, ["pregnancy", "kidney_disease"]),
    "Amlodipine": D("Blood pressure & heart", "ccb", 6),
    "Atenolol": D("Blood pressure & heart", "beta_blocker", 12, ["asthma", "pregnancy"]),
    "Propranolol": D("Blood pressure & heart", "beta_blocker", 12, ["asthma"]),
    "Hydrochlorothiazide": D("Blood pressure & heart", "diuretic", 12, ["gout"]),
    "Furosemide": D("Blood pressure & heart", "diuretic", 1, ["gout"]),
    "Spironolactone": D("Blood pressure & heart", "potassium_sparing", 12, ["kidney_disease", "pregnancy"]),
    # Acid reflux
    "Omeprazole": D("Acid reflux", "ppi", 1),
    "Famotidine": D("Acid reflux", "h2_blocker", 1, ["kidney_disease"]),
    # Allergy and cold
    "Cetirizine": D("Allergy & cold", "antihistamine", 2),
    "Loratadine": D("Allergy & cold", "antihistamine", 2),
    "Chlorpheniramine": D("Allergy & cold", "antihistamine_old", 2, ["glaucoma", "enlarged_prostate"]),
    "Pseudoephedrine": D("Allergy & cold", "decongestant", 12, ["hypertension", "heart_failure", "glaucoma", "hyperthyroidism", "enlarged_prostate"]),
    # Mental health
    "Sertraline": D("Mental health", "ssri", 6),
    "Escitalopram": D("Mental health", "ssri", 12),
    "Amitriptyline": D("Mental health", "tricyclic", 12, ["glaucoma", "enlarged_prostate", "epilepsy", "heart_failure"]),
    "Diazepam": D("Mental health", "benzodiazepine", 6, ["glaucoma", "liver_disease", "pregnancy"]),
    # Asthma
    "Salbutamol": D("Asthma", "bronchodilator"),
    "Montelukast": D("Asthma", "leukotriene", 2),
    # Epilepsy
    "Carbamazepine": D("Epilepsy", "anticonvulsant", 6, ["pregnancy", "liver_disease"]),
    "Valproate": D("Epilepsy", "anticonvulsant", 2, ["pregnancy", "liver_disease"]),
    "Levetiracetam": D("Epilepsy", "anticonvulsant", 1),
    # Thyroid / steroid / gout / nausea
    "Levothyroxine": D("Thyroid", "thyroid_hormone", 0, ["hyperthyroidism"]),
    "Prednisolone": D("Steroid", "corticosteroid", 0, ["diabetes", "peptic_ulcer"]),
    "Allopurinol": D("Gout", "gout_drug"),
    "Colchicine": D("Gout", "gout_drug", 0, ["kidney_disease", "liver_disease"]),
    "Ondansetron": D("Nausea", "antiemetic", 2),
}

CLASSES = sorted(set(d["cls"] for d in DRUGS.values()))

# ------------------------------------------------------------------ interactions
INTERACTIONS = {}


def _add(a, b, msg):
    INTERACTIONS[(a, b)] = msg
    INTERACTIONS[(b, a)] = msg


_add("nsaid", "anticoagulant", "NSAID + anticoagulant raises bleeding risk")
_add("aspirin", "anticoagulant", "Aspirin + anticoagulant raises bleeding risk")
_add("nsaid", "nsaid", "Duplicate NSAID therapy")
_add("nsaid", "aspirin", "NSAID + aspirin: duplicate therapy and stomach bleeding")
_add("nsaid", "ace_inhibitor", "NSAID + ACE inhibitor can harm the kidneys")
_add("nsaid", "arb", "NSAID + ARB can harm the kidneys")
_add("ace_inhibitor", "arb", "ACE inhibitor + ARB: dual blockade")
_add("opioid", "ssri", "Opioid + SSRI: risk of serotonin syndrome")
_add("opioid", "opioid", "Duplicate opioid therapy")
_add("ssri", "nsaid", "SSRI + NSAID raises bleeding risk")
_add("ssri", "aspirin", "SSRI + aspirin raises bleeding risk")
_add("fluoroquinolone", "antidiabetic", "Fluoroquinolone + antidiabetic can cause blood sugar swings")
_add("sulfa", "anticoagulant", "Sulfa antibiotic increases anticoagulant effect")
_add("sulfa", "antidiabetic", "Sulfa antibiotic + antidiabetic can cause low blood sugar")
_add("macrolide", "statin", "Macrolide + statin raises muscle toxicity risk")
_add("macrolide", "anticoagulant", "Macrolide increases anticoagulant effect")
_add("corticosteroid", "nsaid", "Steroid + NSAID raises stomach bleeding risk")
_add("corticosteroid", "antidiabetic", "Steroids raise blood sugar and weaken antidiabetic drugs")
_add("antiplatelet", "nsaid", "Clopidogrel + NSAID raises bleeding risk")
_add("antiplatelet", "aspirin", "Clopidogrel + aspirin raises bleeding risk")
_add("antiplatelet", "anticoagulant", "Clopidogrel + anticoagulant raises bleeding risk")
_add("antiplatelet", "ppi", "Omeprazole can reduce clopidogrel's effect")
_add("antiplatelet", "ssri", "Clopidogrel + SSRI raises bleeding risk")
_add("benzodiazepine", "opioid", "Benzodiazepine + opioid: dangerous breathing depression")
_add("tricyclic", "ssri", "Tricyclic + SSRI: serotonin syndrome risk")
_add("tricyclic", "opioid", "Tricyclic + opioid: serotonin and sedation risk")
_add("nitroimidazole", "anticoagulant", "Metronidazole increases anticoagulant effect")
_add("azole", "anticoagulant", "Azole antifungal increases anticoagulant effect")
_add("azole", "statin", "Azole antifungal + statin raises muscle toxicity risk")
_add("anticonvulsant", "anticoagulant", "Anticonvulsant changes anticoagulant effect")
_add("anticonvulsant", "macrolide", "Macrolide raises anticonvulsant levels")
_add("potassium_sparing", "ace_inhibitor", "Risk of dangerously high potassium")
_add("potassium_sparing", "arb", "Risk of dangerously high potassium")
_add("potassium_sparing", "nsaid", "NSAID + potassium-sparing diuretic: kidney and potassium risk")
_add("diuretic", "nsaid", "NSAID weakens diuretic effect and strains the kidneys")
_add("thyroid_hormone", "anticoagulant", "Levothyroxine increases anticoagulant effect")
_add("beta_blocker", "antidiabetic", "Beta-blocker can hide symptoms of low blood sugar")
_add("gout_drug", "statin", "Colchicine + statin raises muscle toxicity risk")
_add("gout_drug", "macrolide", "Macrolide + colchicine can cause colchicine toxicity")
_add("antihistamine_old", "benzodiazepine", "Additive sedation")
_add("antihistamine_old", "opioid", "Additive sedation")


# ------------------------------------------------------------------ safety rules
def allergy_hit(allergies, drug_cls):
    for a in allergies:
        if drug_cls in CROSS.get(a, [a]):
            return a
    return None


def check_safety_detailed(p, drug):
    # Returns a list of (category, message). Empty list = no problem found.
    d = DRUGS[drug]
    out = []
    hit = allergy_hit(p["allergies"], d["cls"])
    if hit:
        out.append(("Allergies", f"Allergy conflict: patient is allergic to {ALLERGY_LABELS.get(hit, hit).lower()}"))
    for m in p["current_meds"]:
        if m in DRUGS:
            msg = INTERACTIONS.get((d["cls"], DRUGS[m]["cls"]))
            if msg:
                out.append(("Drug interactions", f"Interaction with {m}: {msg}"))
    for c in p["conditions"]:
        if c in d["avoid"]:
            out.append(("Medical conditions", f"Not advised in {CONDITIONS[c].lower()}"))
    if p["age"] < d["min_age"]:
        out.append(("Age", f"Below the minimum age ({d['min_age']} years)"))
    return out


def check_safety(p, drug):
    return [m for _, m in check_safety_detailed(p, drug)]


# ------------------------------------------------------------------ features
def build_features(p, drug):
    d = DRUGS[drug]
    f = {
        "age": p["age"],
        "is_female": int(p["gender"] == "Female"),
        "n_conditions": len(p["conditions"]),
        "n_allergies": len(p["allergies"]),
        "n_current_meds": len(p["current_meds"]),
    }
    for c in CONDITIONS:
        f["cond_" + c] = int(c in p["conditions"])
    for a in ALLERGY_OPTIONS:
        f["allergy_" + a] = int(a in p["allergies"])
    cur = [DRUGS[m]["cls"] for m in p["current_meds"] if m in DRUGS]
    for c in CLASSES:
        f["on_" + c] = cur.count(c)
    for n in DRUGS:
        f["rx_" + n] = int(n == drug)
    f["allergy_conflict"] = int(allergy_hit(p["allergies"], d["cls"]) is not None)
    f["interaction_count"] = sum(1 for c in cur if (d["cls"], c) in INTERACTIONS)
    f["age_below_min"] = int(p["age"] < d["min_age"])
    f["condition_risk_count"] = sum(1 for c in p["conditions"] if c in d["avoid"])
    return f


def row_to_patient(row):
    split = lambda s: [x for x in str(s).split(";") if x and x != "nan"]
    return dict(age=int(row["age"]), gender=row["gender"], conditions=split(row["conditions"]),
                allergies=split(row["allergies"]), current_meds=split(row["previous_medicines"]))


# ------------------------------------------------------------------ synthetic dataset
def _condition_probs(age, gender):
    old = age >= 50
    return {
        "kidney_disease": 0.12 if old else 0.04,
        "liver_disease": 0.06,
        "diabetes": 0.22 if old else 0.05,
        "hypertension": 0.35 if old else 0.07,
        "heart_failure": 0.08 if old else 0.01,
        "asthma": 0.09,
        "peptic_ulcer": 0.06,
        "epilepsy": 0.02,
        "glaucoma": 0.05 if old else 0.01,
        "gout": 0.08 if old else 0.02,
        "bleeding_disorder": 0.02,
        "g6pd_deficiency": 0.03,
        "hyperthyroidism": 0.03,
        "enlarged_prostate": 0.15 if (old and gender == "Male") else 0.0,
    }


def generate_dataset(n=12000, seed=42):
    rnd = random.Random(seed)
    names = list(DRUGS)
    rows = []
    for i in range(n):
        age = int(min(95, max(1, rnd.gauss(42, 22))))
        gender = rnd.choice(["Male", "Female"])
        conds = [c for c, pr in _condition_probs(age, gender).items() if rnd.random() < pr]
        if gender == "Female" and 16 <= age <= 45 and rnd.random() < 0.10:
            conds.append("pregnancy")
        allergies = [a for a in ALLERGY_OPTIONS if rnd.random() < 0.05]
        k = rnd.choices([0, 1, 2, 3, 4], weights=[25, 30, 25, 15, 5])[0]
        meds = rnd.sample(names, k)
        drug = rnd.choice(names)
        p = dict(age=age, gender=gender, conditions=conds, allergies=allergies, current_meds=meds)
        reasons = check_safety(p, drug)
        rows.append(dict(patient_id=f"P{i + 1:05d}", age=age, gender=gender,
                         conditions=";".join(conds), allergies=";".join(allergies),
                         previous_medicines=";".join(meds), prescribed_medicine=drug,
                         is_safe=int(len(reasons) == 0),
                         reason=" | ".join(reasons) if reasons else "No issues found"))
    return pd.DataFrame(rows)
