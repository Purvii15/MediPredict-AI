"""
medical_rules.py
================
Single source of truth for:
  - Symptom importance weights  (1 = generic, 5 = highly specific/severe)
  - Disease severity tiers      (low / medium / high)
  - Critical symptom gates      (high-severity diseases need these to appear)
  - Disease severity tags       (emoji + label shown in UI)
"""

# ══════════════════════════════════════════════════════════════════════════════
# SYMPTOM WEIGHTS
# Scale: 1 (very common, non-specific) → 5 (rare, highly disease-specific)
# Used to build weighted feature vectors for model training AND
# to compute a "total symptom weight score" at prediction time.
# ══════════════════════════════════════════════════════════════════════════════
SYMPTOM_WEIGHTS = {
    # ── Weight 1: extremely common, non-specific ──────────────────────────
    "fatigue":                      1,
    "headache":                     1,
    "nausea":                       1,
    "vomiting":                     1,
    "loss_of_appetite":             1,
    "mild_fever":                   1,
    "sweating":                     1,
    "chills":                       1,
    "shivering":                    1,
    "malaise":                      1,
    "restlessness":                 1,
    "lethargy":                     1,
    "anxiety":                      1,
    "mood_swings":                  1,
    "weight_loss":                  1,
    "weight_gain":                  1,
    "back_pain":                    1,
    "constipation":                 1,
    "diarrhoea":                    1,
    "indigestion":                  1,
    "abdominal_pain":               1,
    "stomach_pain":                 1,
    "dehydration":                  1,
    "runny_nose":                   1,
    "congestion":                   1,
    "sneezing":                     1,
    "continuous_sneezing":          1,
    "cough":                        1,
    "throat_irritation":            1,
    "skin_rash":                    1,
    "itching":                      1,
    "muscle_pain":                  1,
    "joint_pain":                   1,
    "dizziness":                    1,
    "depression":                   1,
    "irritability":                 1,

    # ── Weight 2: moderately specific ────────────────────────────────────
    "high_fever":                   2,
    "breathlessness":               2,
    "chest_pain":                   2,
    "fast_heart_rate":              2,
    "palpitations":                 2,
    "yellowish_skin":               2,
    "dark_urine":                   2,
    "yellow_urine":                 2,
    "yellowing_of_eyes":            2,
    "phlegm":                       2,
    "mucoid_sputum":                2,
    "rusty_sputum":                 2,
    "blood_in_sputum":              2,
    "neck_pain":                    2,
    "stiff_neck":                   2,
    "knee_pain":                    2,
    "hip_joint_pain":               2,
    "swelling_joints":              2,
    "movement_stiffness":           2,
    "muscle_weakness":              2,
    "burning_micturition":          2,
    "bladder_discomfort":           2,
    "continuous_feel_of_urine":     2,
    "polyuria":                     2,
    "irregular_sugar_level":        2,
    "increased_appetite":           2,
    "excessive_hunger":             2,
    "blurred_and_distorted_vision": 2,
    "visual_disturbances":          2,
    "sunken_eyes":                  2,
    "puffy_face_and_eyes":          2,
    "enlarged_thyroid":             2,
    "brittle_nails":                2,
    "swollen_extremeties":          2,
    "cold_hands_and_feets":         2,
    "obesity":                      2,
    "swollen_legs":                 2,
    "prominent_veins_on_calf":      2,
    "painful_walking":              2,
    "spinning_movements":           2,
    "loss_of_balance":              2,
    "unsteadiness":                 2,
    "loss_of_smell":                2,
    "patches_in_throat":            2,
    "nodal_skin_eruptions":         2,
    "dischromic_patches":           2,
    "red_spots_over_body":          2,
    "pus_filled_pimples":           2,
    "blackheads":                   2,
    "skin_peeling":                 2,
    "silver_like_dusting":          2,
    "small_dents_in_nails":         2,
    "inflammatory_nails":           2,
    "blister":                      2,
    "red_sore_around_nose":         2,
    "yellow_crust_ooze":            2,
    "scurring":                     2,
    "watering_from_eyes":           2,
    "redness_of_eyes":              2,
    "sinus_pressure":               2,
    "pain_behind_the_eyes":         2,
    "passage_of_gases":             2,
    "belly_pain":                   2,
    "internal_itching":             2,
    "acidity":                      2,
    "ulcers_on_tongue":             2,
    "pain_during_bowel_movements":  2,
    "pain_in_anal_region":          2,
    "bloody_stool":                 2,
    "irritation_in_anus":           2,
    "bruising":                     2,
    "swelled_lymph_nodes":          2,
    "abnormal_menstruation":        2,
    "spotting_urination":           2,
    "family_history":               2,
    "lack_of_concentration":        2,
    "drying_and_tingling_lips":     2,

    # ── Weight 3: fairly specific ─────────────────────────────────────────
    "acute_liver_failure":          3,
    "fluid_overload":               3,
    "swelling_of_stomach":          3,
    "distention_of_abdomen":        3,
    "stomach_bleeding":             3,
    "history_of_alcohol_consumption": 3,
    "receiving_blood_transfusion":  3,
    "receiving_unsterile_injections": 3,
    "altered_sensorium":            3,
    "toxic_look_(typhos)":          3,
    "extra_marital_contacts":       3,
    "muscle_wasting":               3,
    "hip_joint_pain":               3,
    "swollen_blood_vessels":        3,

    # ── Weight 4: highly specific, serious ───────────────────────────────
    "weakness_of_one_body_side":    4,
    "slurred_speech":               4,
    "loss_of_smell":                4,
    "coma":                         4,

    # ── Weight 5: critical / emergency indicators ─────────────────────────
    "paralysis":                    5,   # conceptual — maps to weakness_of_one_body_side
    "weakness_in_limbs":            5,
}

DEFAULT_WEIGHT = 1  # fallback for any symptom not listed above


def get_weight(symptom: str) -> int:
    return SYMPTOM_WEIGHTS.get(symptom, DEFAULT_WEIGHT)


def total_weight(symptoms: list) -> int:
    """Sum of weights for a list of symptom names."""
    return sum(get_weight(s) for s in symptoms)


# ══════════════════════════════════════════════════════════════════════════════
# DISEASE SEVERITY TIERS
# ══════════════════════════════════════════════════════════════════════════════
SEVERITY = {
    # ── LOW — common, self-limiting ───────────────────────────────────────
    "Fungal infection":                     "low",
    "Allergy":                              "low",
    "Common Cold":                          "low",
    "Acne":                                 "low",
    "Impetigo":                             "low",
    "Chicken pox":                          "low",
    "Drug Reaction":                        "low",

    # ── MEDIUM — needs medical attention but not emergency ────────────────
    "GERD":                                 "medium",
    "Chronic cholestasis":                  "medium",
    "Peptic ulcer diseae":                  "medium",
    "Gastroenteritis":                      "medium",
    "Urinary tract infection":              "medium",
    "Psoriasis":                            "medium",
    "Dimorphic hemmorhoids(piles)":         "medium",
    "Varicose veins":                       "medium",
    "Cervical spondylosis":                 "medium",
    "Osteoarthristis":                      "medium",
    "Arthritis":                            "medium",
    "Hypothyroidism":                       "medium",
    "Hyperthyroidism":                      "medium",
    "Hypoglycemia":                         "medium",
    "Hypertension":                         "medium",
    "Migraine":                             "medium",
    "Bronchial Asthma":                     "medium",
    "Jaundice":                             "medium",
    "Malaria":                              "medium",
    "Dengue":                               "medium",
    "Typhoid":                              "medium",
    "Tuberculosis":                         "medium",
    "Pneumonia":                            "medium",
    "hepatitis A":                          "medium",
    "Hepatitis E":                          "medium",
    "Diabetes":                             "medium",
    "(vertigo) Paroymsal  Positional Vertigo": "medium",

    # ── HIGH — serious / potentially life-threatening ─────────────────────
    "AIDS":                                 "high",
    "Hepatitis B":                          "high",
    "Hepatitis C":                          "high",
    "Hepatitis D":                          "high",
    "Alcoholic hepatitis":                  "high",
    "Heart attack":                         "high",
    "Paralysis (brain hemorrhage)":         "high",
}

SEVERITY_UI = {
    "low":    {"emoji": "🟢", "label": "Mild",     "css": "sev-low"},
    "medium": {"emoji": "🟡", "label": "Moderate", "css": "sev-medium"},
    "high":   {"emoji": "🔴", "label": "Serious",  "css": "sev-high"},
}


def get_severity(disease: str) -> str:
    return SEVERITY.get(disease, "medium")


def get_severity_ui(disease: str) -> dict:
    return SEVERITY_UI[get_severity(disease)]


# ══════════════════════════════════════════════════════════════════════════════
# CRITICAL SYMPTOM GATES
# High-severity diseases require at least MIN_CRITICAL of these symptoms.
# If not present → probability is penalised heavily.
# ══════════════════════════════════════════════════════════════════════════════
CRITICAL_GATES = {
    "Paralysis (brain hemorrhage)": {
        "required": ["weakness_of_one_body_side", "slurred_speech",
                     "weakness_in_limbs", "altered_sensorium", "loss_of_balance"],
        "min_match": 2,
    },
    "Heart attack": {
        "required": ["chest_pain", "fast_heart_rate", "palpitations",
                     "breathlessness", "sweating"],
        "min_match": 2,
    },
    "AIDS": {
        "required": ["receiving_blood_transfusion", "receiving_unsterile_injections",
                     "extra_marital_contacts", "muscle_wasting"],
        "min_match": 1,
    },
    "Hepatitis B": {
        "required": ["receiving_blood_transfusion", "receiving_unsterile_injections",
                     "yellowish_skin", "dark_urine", "yellowing_of_eyes"],
        "min_match": 2,
    },
    "Hepatitis C": {
        "required": ["receiving_blood_transfusion", "receiving_unsterile_injections",
                     "yellowish_skin", "dark_urine"],
        "min_match": 1,
    },
    "Hepatitis D": {
        "required": ["receiving_blood_transfusion", "receiving_unsterile_injections",
                     "yellowish_skin", "dark_urine"],
        "min_match": 1,
    },
    "Alcoholic hepatitis": {
        "required": ["history_of_alcohol_consumption", "yellowish_skin",
                     "fluid_overload", "stomach_bleeding"],
        "min_match": 1,
    },
}

# Minimum total symptom weight for a high-severity disease to appear at all
HIGH_SEVERITY_MIN_WEIGHT = 6


def passes_gate(disease: str, selected_symptoms: list) -> tuple:
    """
    Returns (passes: bool, matched_critical: int, required_critical: int).
    - If disease has no gate → always passes.
    - If disease is high severity but total weight is too low → fails.
    """
    selected = set(selected_symptoms)
    sev = get_severity(disease)

    # Weight gate for ALL high-severity diseases
    if sev == "high":
        tw = total_weight(selected_symptoms)
        if tw < HIGH_SEVERITY_MIN_WEIGHT:
            return False, 0, 0

    gate = CRITICAL_GATES.get(disease)
    if gate is None:
        return True, 0, 0

    matched = len(selected & set(gate["required"]))
    passes  = matched >= gate["min_match"]
    return passes, matched, gate["min_match"]


# ══════════════════════════════════════════════════════════════════════════════
# PENALTY FACTOR
# Applied to RF probability when a disease fails its gate.
# ══════════════════════════════════════════════════════════════════════════════
GATE_PENALTY = 0.05   # multiply raw proba by this if gate fails
