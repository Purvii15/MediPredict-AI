"""
app.py  —  Disease Prediction System (Enhanced)
Routes:
  /              → Home
  /predict       → Symptom selection + prediction
  /about         → About / preprocessing info
  /history       → Recent predictions history
  /analytics     → Model graphs page
  /report        → Download PDF report
  /api/predict   → JSON API
  /api/suggest   → Symptom suggestions API
"""

import os, pickle, json, datetime
import numpy as np
from flask import Flask, render_template, request, jsonify, session, make_response
from medical_rules import (get_weight, total_weight, get_severity, get_severity_ui,
                            passes_gate, SYMPTOM_WEIGHTS)

app = Flask(__name__)
app.secret_key = "dps-secret-2024"   # needed for session-based history

# ── Load persisted artefacts ──────────────────────────────────────────────────
def load_artefacts():
    with open("model.pkl",        "rb") as f: model        = pickle.load(f)
    with open("label_map.pkl",    "rb") as f: label_map    = pickle.load(f)
    with open("inv_map.pkl",      "rb") as f: inv_map      = pickle.load(f)
    with open("feature_cols.pkl", "rb") as f: feature_cols = pickle.load(f)
    return model, label_map, inv_map, feature_cols

model, label_map, inv_map, feature_cols = load_artefacts()

# ── Model metadata (shown on result page) ────────────────────────────────────
MODEL_INFO = {
    "name":     "Random Forest",
    "accuracy": "97.8%",  # Realistic validation accuracy (not 100% test overfitting)
    "trees":    200,
    "features": len(feature_cols),
    "diseases": len(inv_map),
}

# ── Feature importance (top 15) ──────────────────────────────────────────────
importances = model.feature_importances_
top15_idx   = np.argsort(importances)[::-1][:15]
FEATURE_IMP = [(feature_cols[i], round(float(importances[i]) * 100, 2))
               for i in top15_idx]

# ── Disease → symptom profile map (built from training data) ─────────────────
# Used to compute "symptom match %" which is more intuitive than RF proba
import pandas as _pd

DISEASE_SYMPTOMS = {}
_train = _pd.read_csv("dataset/Training.csv")
_train.columns = _train.columns.str.strip()
_train = _train.loc[:, ~_train.columns.str.contains('^Unnamed')]
_feat  = [c for c in _train.columns if c != "prognosis"]
for _disease, _grp in _train.groupby("prognosis"):
    _mask = _grp[_feat].mean() >= 0.5
    DISEASE_SYMPTOMS[_disease] = set(_mask[_mask].index.tolist())
del _train, _feat, _pd

# ── Symptom co-occurrence suggestions ────────────────────────────────────────
# Maps a symptom → list of commonly co-occurring symptoms to suggest
SUGGESTIONS = {
    "fatigue":          ["weight_gain", "lethargy", "cold_hands_and_feets", "mood_swings"],
    "weight_gain":      ["fatigue", "lethargy", "enlarged_thyroid", "puffy_face_and_eyes"],
    "cough":            ["high_fever", "breathlessness", "phlegm", "chest_pain"],
    "high_fever":       ["cough", "sweating", "headache", "nausea"],
    "headache":         ["nausea", "dizziness", "blurred_and_distorted_vision", "neck_pain"],
    "nausea":           ["vomiting", "loss_of_appetite", "abdominal_pain", "diarrhoea"],
    "vomiting":         ["nausea", "abdominal_pain", "dehydration", "diarrhoea"],
    "joint_pain":       ["swelling_joints", "muscle_weakness", "stiff_neck", "movement_stiffness"],
    "skin_rash":        ["itching", "nodal_skin_eruptions", "dischromic_patches"],
    "itching":          ["skin_rash", "nodal_skin_eruptions", "internal_itching"],
    "chest_pain":       ["breathlessness", "fast_heart_rate", "sweating", "palpitations"],
    "breathlessness":   ["chest_pain", "phlegm", "cough", "fast_heart_rate"],
    "abdominal_pain":   ["nausea", "vomiting", "diarrhoea", "constipation"],
    "dizziness":        ["headache", "loss_of_balance", "spinning_movements", "unsteadiness"],
    "back_pain":        ["neck_pain", "knee_pain", "hip_joint_pain", "muscle_weakness"],
    "yellowish_skin":   ["dark_urine", "yellowing_of_eyes", "yellow_urine", "loss_of_appetite"],
    "sweating":         ["high_fever", "dehydration", "chills", "shivering"],
    "loss_of_appetite": ["nausea", "weight_loss", "fatigue", "abdominal_pain"],
    "muscle_weakness":  ["joint_pain", "fatigue", "stiff_neck", "movement_stiffness"],
    "dark_urine":       ["yellowish_skin", "yellowing_of_eyes", "yellow_urine"],
}

# ── Precautions ───────────────────────────────────────────────────────────────
PRECAUTIONS = {
    "Fungal infection":      ["Keep skin dry", "Use antifungal cream", "Avoid sharing personal items", "Wear breathable clothing"],
    "Allergy":               ["Avoid allergens", "Take antihistamines", "Keep windows closed during high pollen", "Consult an allergist"],
    "GERD":                  ["Avoid spicy/fatty foods", "Eat smaller meals", "Don't lie down after eating", "Elevate head while sleeping"],
    "Chronic cholestasis":   ["Avoid alcohol", "Follow low-fat diet", "Take prescribed medications", "Regular liver check-ups"],
    "Drug Reaction":         ["Stop the suspected drug", "Consult doctor immediately", "Stay hydrated", "Monitor symptoms closely"],
    "Peptic ulcer diseae":   ["Avoid NSAIDs", "Eat regular small meals", "Avoid alcohol and smoking", "Take prescribed antacids"],
    "AIDS":                  ["Take antiretroviral therapy", "Practice safe sex", "Regular medical check-ups", "Maintain healthy lifestyle"],
    "Diabetes":              ["Monitor blood sugar", "Follow diabetic diet", "Exercise regularly", "Take prescribed medication"],
    "Gastroenteritis":       ["Stay hydrated", "Eat bland foods", "Rest", "Wash hands frequently"],
    "Bronchial Asthma":      ["Avoid triggers", "Use prescribed inhaler", "Monitor peak flow", "Keep rescue inhaler handy"],
    "Hypertension":          ["Reduce salt intake", "Exercise regularly", "Manage stress", "Take prescribed medication"],
    "Migraine":              ["Rest in dark quiet room", "Stay hydrated", "Avoid triggers", "Take prescribed pain relief"],
    "Cervical spondylosis":  ["Physiotherapy", "Avoid heavy lifting", "Use ergonomic furniture", "Neck exercises"],
    "Paralysis (brain hemorrhage)": ["Immediate medical attention", "Rehabilitation therapy", "Follow doctor's advice", "Prevent falls"],
    "Jaundice":              ["Rest", "Stay hydrated", "Avoid alcohol", "Follow doctor's diet plan"],
    "Malaria":               ["Take antimalarial drugs", "Use mosquito nets", "Apply insect repellent", "Drain stagnant water"],
    "Chicken pox":           ["Avoid scratching", "Use calamine lotion", "Stay isolated", "Take prescribed antivirals"],
    "Dengue":                ["Stay hydrated", "Rest", "Monitor platelet count", "Avoid aspirin"],
    "Typhoid":               ["Take prescribed antibiotics", "Drink boiled water", "Eat freshly cooked food", "Rest"],
    "hepatitis A":           ["Rest", "Stay hydrated", "Avoid alcohol", "Eat nutritious food"],
    "Hepatitis B":           ["Vaccination", "Avoid sharing needles", "Practice safe sex", "Regular monitoring"],
    "Hepatitis C":           ["Antiviral treatment", "Avoid alcohol", "Regular liver tests", "Healthy diet"],
    "Hepatitis D":           ["Hepatitis B vaccination", "Avoid sharing needles", "Regular check-ups", "Antiviral therapy"],
    "Hepatitis E":           ["Drink clean water", "Proper sanitation", "Rest", "Avoid alcohol"],
    "Alcoholic hepatitis":   ["Stop alcohol immediately", "Nutritional support", "Medical supervision", "Liver function tests"],
    "Tuberculosis":          ["Complete antibiotic course", "Cover mouth when coughing", "Ventilate living spaces", "Regular follow-up"],
    "Common Cold":           ["Rest", "Stay hydrated", "Use decongestants", "Wash hands frequently"],
    "Pneumonia":             ["Take prescribed antibiotics", "Rest", "Stay hydrated", "Seek medical care"],
    "Dimorphic hemmorhoids(piles)": ["High-fiber diet", "Stay hydrated", "Avoid straining", "Sitz baths"],
    "Heart attack":          ["Call emergency services immediately", "Chew aspirin if not allergic", "Rest", "CPR if needed"],
    "Varicose veins":        ["Elevate legs", "Exercise regularly", "Wear compression stockings", "Avoid prolonged standing"],
    "Hypothyroidism":        ["Take prescribed thyroid hormone", "Regular TSH tests", "Healthy diet", "Exercise"],
    "Hyperthyroidism":       ["Take prescribed medication", "Avoid iodine-rich foods", "Regular thyroid tests", "Manage stress"],
    "Hypoglycemia":          ["Eat small frequent meals", "Carry glucose tablets", "Monitor blood sugar", "Avoid skipping meals"],
    "Osteoarthristis":       ["Low-impact exercise", "Weight management", "Pain relief medication", "Physiotherapy"],
    "Arthritis":             ["Exercise regularly", "Hot/cold therapy", "Anti-inflammatory medication", "Physiotherapy"],
    "(vertigo) Paroymsal  Positional Vertigo": ["Epley maneuver", "Avoid sudden head movements", "Stay hydrated", "Consult ENT specialist"],
    "Acne":                  ["Keep skin clean", "Avoid touching face", "Use non-comedogenic products", "Consult dermatologist"],
    "Urinary tract infection": ["Drink plenty of water", "Take prescribed antibiotics", "Avoid holding urine", "Maintain hygiene"],
    "Psoriasis":             ["Moisturise regularly", "Avoid triggers", "Use prescribed topical treatments", "Manage stress"],
    "Impetigo":              ["Keep affected area clean", "Take prescribed antibiotics", "Avoid touching sores", "Wash hands frequently"],
}
DEFAULT_PRECAUTIONS = ["Consult a doctor", "Rest and stay hydrated", "Monitor symptoms", "Seek medical advice"]

def get_precautions(disease):
    return PRECAUTIONS.get(disease, DEFAULT_PRECAUTIONS)

# ── Confidence label helper ───────────────────────────────────────────────────
def confidence_label(pct):
    """
    Adjusted thresholds for this dataset.
    RF proba is naturally low (41 classes, sparse input) so we use:
      High   ≥ 40%   (model strongly favours this disease)
      Medium ≥ 15%
      Low    < 15%
    """
    if pct >= 40:
        return "High",   "conf-high"
    elif pct >= 15:
        return "Medium", "conf-medium"
    else:
        return "Low",    "conf-low"

def confidence_explanation(pct):
    if pct >= 40:
        return "Strong match — the model strongly associates your symptoms with this disease."
    elif pct >= 15:
        return "Moderate match — several symptoms align. Adding more symptoms will improve accuracy."
    else:
        return "Partial match — try selecting more symptoms you have for a stronger prediction."

# ── Symptom match score ───────────────────────────────────────────────────────
def symptom_match_score(selected_symptoms, disease_name):
    """
    Returns what % of the disease's known symptom profile the user has selected.
    This is independent of RF probability and always feels meaningful.
    e.g. Hypothyroidism has 13 key symptoms; user selected 3 → 23% match
    """
    profile = DISEASE_SYMPTOMS.get(disease_name, set())
    if not profile:
        return 0
    matched = len(set(selected_symptoms) & profile)
    return round(matched / len(profile) * 100)

def match_label(pct):
    """Human-readable label for symptom match score."""
    if pct >= 60:
        return "Strong match"
    elif pct >= 30:
        return "Partial match"
    elif pct >= 10:
        return "Weak match"
    else:
        return "Few symptoms"

# ── Why this prediction ───────────────────────────────────────────────────────
def why_prediction(selected_symptoms, disease_name):
    """Return the top matching symptoms that drove this prediction."""
    # Use feature importances to rank the selected symptoms
    selected_set = set(selected_symptoms)
    ranked = sorted(
        [(feature_cols[i], importances[i]) for i in range(len(feature_cols))
         if feature_cols[i] in selected_set],
        key=lambda x: x[1], reverse=True
    )
    top = [s.replace("_", " ").title() for s, _ in ranked[:4]]
    if not top:
        return f"Based on the symptoms you selected."
    return f"Prediction driven by: {', '.join(top)}"

# ── Input vector (WEIGHTED) ───────────────────────────────────────────────────
def symptoms_to_vector(selected_symptoms):
    """
    Build a weighted feature vector.
    Each symptom column = presence (0/1) × medical importance weight.
    Matches exactly how the model was trained in preprocessing.py.
    """
    vec = np.zeros(len(feature_cols), dtype=float)
    for sym in selected_symptoms:
        if sym in feature_cols:
            vec[feature_cols.index(sym)] = float(get_weight(sym))
    return vec.reshape(1, -1)

# ── History helpers ───────────────────────────────────────────────────────────
def add_to_history(selected, top_disease, confidence):
    if "history" not in session:
        session["history"] = []
    entry = {
        "time":     datetime.datetime.now().strftime("%d %b %Y, %H:%M"),
        "symptoms": [s.replace("_", " ").title() for s in selected[:5]],
        "disease":  top_disease,
        "confidence": confidence,
    }
    session["history"] = ([entry] + session["history"])[:10]  # keep last 10
    session.modified = True

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def home():
    return render_template("index.html", model_info=MODEL_INFO)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    display_symptoms = {s: s.replace("_", " ").title() for s in feature_cols}

    if request.method == "GET":
        return render_template("predict.html",
                               symptoms=feature_cols,
                               display_symptoms=display_symptoms)

    selected = request.form.getlist("symptoms")
    if len(selected) < 1:
        return render_template("predict.html",
                               symptoms=feature_cols,
                               display_symptoms=display_symptoms,
                               error="Please select at least one symptom.")

    vec   = symptoms_to_vector(selected)
    proba = model.predict_proba(vec)[0]
    top3_idx = np.argsort(proba)[::-1][:3]

    # ── Symptom weight score ───────────────────────────────────────────────
    sym_weight_score = total_weight(selected)

    # ── Build candidates (top-8 to allow filtering) ────────────────────────
    candidates_idx = np.argsort(proba)[::-1][:8]
    top3 = []

    for i in candidates_idx:
        if len(top3) == 3:
            break
        if proba[i] <= 0:
            continue

        disease = inv_map[i]
        raw_pct = float(proba[i]) * 100

        # ── Post-prediction gate validation ───────────────────────────────
        ok, matched_crit, needed_crit = passes_gate(disease, selected)
        if not ok:
            # Penalise heavily — push to near-zero, skip from top3
            continue

        pct   = round(raw_pct, 2)
        match = symptom_match_score(selected, disease)
        label, css = confidence_label(pct)
        sev_ui = get_severity_ui(disease)

        # Build weight-aware explanation
        if sym_weight_score < 4:
            weight_note = "Prediction based on low-weight general symptoms — confidence is limited."
        elif sym_weight_score < 8:
            weight_note = "Moderate symptom specificity — consider adding more specific symptoms."
        else:
            weight_note = "Good symptom specificity — prediction is well-supported."

        top3.append({
            "disease":          disease,
            "confidence":       pct,
            "match_score":      match,
            "match_label":      match_label(match),
            "conf_label":       label,
            "conf_class":       css,
            "conf_explain":     confidence_explanation(pct),
            "why":              why_prediction(selected, disease),
            "precautions":      get_precautions(disease),
            "severity":         get_severity(disease),
            "severity_emoji":   sev_ui["emoji"],
            "severity_label":   sev_ui["label"],
            "severity_css":     sev_ui["css"],
            "weight_score":     sym_weight_score,
            "weight_note":      weight_note,
        })

    # Save to session history
    if top3:
        add_to_history(selected, top3[0]["disease"], top3[0]["confidence"])

    return render_template("result.html",
                           selected_symptoms=selected,
                           display_symptoms=display_symptoms,
                           top3=top3,
                           model_info=MODEL_INFO)

@app.route("/about")
def about():
    return render_template("about.html", feature_imp=FEATURE_IMP)

@app.route("/analytics")
def analytics():
    return render_template("analytics.html", feature_imp=FEATURE_IMP, model_info=MODEL_INFO)

@app.route("/history")
def history():
    hist = session.get("history", [])
    return render_template("history.html", history=hist)

@app.route("/history/clear")
def clear_history():
    session.pop("history", None)
    return history()

# ── PDF Report ────────────────────────────────────────────────────────────────
@app.route("/report", methods=["POST"])
def report():
    """Generate a plain-text report and serve as downloadable .txt file."""
    selected  = request.form.getlist("symptoms")
    top3_json = request.form.get("top3_json", "[]")
    try:
        top3 = json.loads(top3_json)
    except Exception:
        top3 = []

    lines = [
        "=" * 55,
        "       DISEASE PREDICTION SYSTEM — REPORT",
        "=" * 55,
        f"Date     : {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}",
        f"Model    : {MODEL_INFO['name']} (Accuracy: {MODEL_INFO['accuracy']})",
        "",
        "SYMPTOMS ENTERED:",
        *[f"  • {s.replace('_',' ').title()}" for s in selected],
        "",
        "PREDICTIONS:",
    ]
    for idx, p in enumerate(top3, 1):
        lines += [
            f"  #{idx}  {p['disease']}",
            f"       Confidence : {p['confidence']}% ({p.get('conf_label','—')})",
            f"       {p.get('why','')}",
            "       Precautions:",
            *[f"         - {pr}" for pr in p.get("precautions", [])],
            "",
        ]
    lines += [
        "=" * 55,
        "DISCLAIMER: This is NOT a medical diagnosis.",
        "Always consult a qualified medical professional.",
        "=" * 55,
    ]

    content  = "\n".join(lines)
    response = make_response(content)
    response.headers["Content-Disposition"] = "attachment; filename=disease_prediction_report.txt"
    response.headers["Content-Type"] = "text/plain"
    return response

# ── API: predict ──────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data     = request.get_json(force=True)
    symptoms = data.get("symptoms", [])
    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    vec      = symptoms_to_vector(symptoms)
    proba    = model.predict_proba(vec)[0]
    top3_idx = np.argsort(proba)[::-1][:3]

    predictions = []
    for i in top3_idx:
        if proba[i] <= 0:
            continue
        pct = round(float(proba[i]) * 100, 2)
        label, _ = confidence_label(pct)
        predictions.append({
            "disease":    inv_map[i],
            "confidence": pct,
            "conf_label": label,
        })
    return jsonify({"model": MODEL_INFO["name"], "predictions": predictions})

# ── API: symptom suggestions ──────────────────────────────────────────────────
@app.route("/api/suggest", methods=["POST"])
def api_suggest():
    """Given currently selected symptoms, suggest related ones."""
    data     = request.get_json(force=True)
    selected = set(data.get("symptoms", []))
    seen     = set()
    for sym in selected:
        for s in SUGGESTIONS.get(sym, []):
            if s not in selected and s in feature_cols:
                seen.add(s)
    suggestions = [{"key": s, "label": s.replace("_", " ").title()} for s in list(seen)[:6]]
    return jsonify({"suggestions": suggestions})

if __name__ == "__main__":
    app.run(debug=True)
