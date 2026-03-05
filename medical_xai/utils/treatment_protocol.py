"""
treatment_protocol.py
=====================
Generates structured treatment recommendations for detected X-ray findings.

Covers:
  - Immediate actions
  - Medications (first-line)
  - Investigations to order
  - Lifestyle / supportive care
  - Follow-up timeline
  - Red flags (when to escalate)

⚠️ For educational/research use only. Not a substitute for clinical judgement.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TreatmentPlan:
    condition: str
    severity:  str
    immediate_actions:  List[str] = field(default_factory=list)
    medications:        List[dict] = field(default_factory=list)
    investigations:     List[str]  = field(default_factory=list)
    lifestyle:          List[str]  = field(default_factory=list)
    follow_up:          List[str]  = field(default_factory=list)
    red_flags:          List[str]  = field(default_factory=list)
    prognosis:          str = ""


# ── Pneumonia treatment protocols by severity ─────────────────────────────

PNEUMONIA_PROTOCOLS = {
    "Severe": TreatmentPlan(
        condition="Bacterial Pneumonia",
        severity="Severe",
        immediate_actions=[
            "URGENT: Hospital admission — consider ICU if SpO2 < 90%",
            "Supplemental oxygen (target SpO2 ≥ 94%)",
            "IV access + fluid resuscitation if hypotensive",
            "Blood cultures × 2 before starting antibiotics",
            "Arterial blood gas (ABG) assessment",
            "Continuous pulse oximetry monitoring",
        ],
        medications=[
            {"name": "Co-Amoxiclav (Augmentin)",  "dose": "1.2 g IV every 8 hours",   "duration": "5–7 days IV, then oral step-down"},
            {"name": "Azithromycin",               "dose": "500 mg IV/oral once daily", "duration": "5 days (atypical cover)"},
            {"name": "Ceftriaxone (if penicillin allergy)", "dose": "2 g IV once daily", "duration": "5–7 days"},
            {"name": "Paracetamol (Acetaminophen)","dose": "1 g every 6 hours",        "duration": "As needed for fever/pain"},
            {"name": "Salbutamol nebulization",    "dose": "2.5 mg every 4–6 hours",   "duration": "While wheezing persists"},
        ],
        investigations=[
            "Chest X-ray (confirm + baseline)",
            "Full blood count (FBC) + CRP + ESR",
            "Renal function tests (U&E, creatinine)",
            "Liver function tests (LFTs)",
            "Blood cultures × 2 (aerobic + anaerobic)",
            "Sputum culture and sensitivity (C&S)",
            "Urine Legionella antigen + Pneumococcal antigen",
            "HIV test if immunocompromised",
            "Procalcitonin (PCT) — severity marker",
        ],
        lifestyle=[
            "Complete bed rest during acute phase",
            "High fluid intake (2–3 L/day unless contraindicated)",
            "Incentive spirometry to prevent atelectasis",
            "Elevate head of bed to 30–45°",
            "Avoid smoking absolutely — no exceptions",
            "No alcohol during antibiotic course",
        ],
        follow_up=[
            "Repeat CXR at 4–6 weeks post-treatment (ensure resolution)",
            "Outpatient review at 1 week after discharge",
            "Pneumococcal vaccine after recovery (if not previously given)",
            "Influenza vaccine annually",
            "Consider CT thorax if CXR doesn't clear by 6 weeks",
        ],
        red_flags=[
            "SpO2 < 92% despite O2 → escalate to ICU",
            "Confusion / altered mental state (CURB-65 score)",
            "Systolic BP < 90 mmHg → septic shock protocol",
            "RR > 30 breaths/min → respiratory failure",
            "Failure to improve within 48–72 hours → change antibiotics",
            "Pleural effusion developing → consider drainage",
        ],
        prognosis="With prompt treatment: good recovery expected in 2–4 weeks. Mortality risk elevated without early intervention."
    ),

    "Moderate": TreatmentPlan(
        condition="Bacterial Pneumonia",
        severity="Moderate",
        immediate_actions=[
            "Hospital admission recommended (CURB-65 score ≥ 2)",
            "Supplemental oxygen if SpO2 < 94%",
            "Oral or IV antibiotics based on severity",
            "Blood cultures if hospitalized",
        ],
        medications=[
            {"name": "Amoxicillin",   "dose": "500 mg–1 g oral TDS",    "duration": "5–7 days"},
            {"name": "Azithromycin",  "dose": "500 mg oral once daily",  "duration": "5 days (if atypical suspected)"},
            {"name": "Doxycycline",   "dose": "200 mg loading, then 100 mg BD", "duration": "5–7 days (alternative)"},
            {"name": "Paracetamol",   "dose": "1 g every 6 hours",       "duration": "As needed"},
            {"name": "Ibuprofen",     "dose": "400 mg TDS with food",    "duration": "3–5 days (if no contraindication)"},
        ],
        investigations=[
            "Chest X-ray (PA + lateral)",
            "FBC, CRP, ESR",
            "Urea and electrolytes",
            "Sputum culture if productive cough",
            "Blood cultures (if febrile / hospitalized)",
            "Pulse oximetry",
        ],
        lifestyle=[
            "Rest at home or hospital depending on CURB-65",
            "Stay well-hydrated (2–3 L/day)",
            "Steam inhalation for congestion relief",
            "Avoid cold environments",
            "No smoking during illness",
            "Nutritious diet rich in Vitamin C and zinc",
        ],
        follow_up=[
            "GP/clinic review in 48 hours if not improving",
            "Repeat CXR at 4–6 weeks post-treatment",
            "Pneumococcal + flu vaccination post-recovery",
        ],
        red_flags=[
            "Worsening breathlessness → seek emergency care",
            "SpO2 dropping below 94%",
            "Not improving after 48–72 hours on antibiotics",
            "New confusion or drowsiness",
        ],
        prognosis="With timely oral antibiotics: expected recovery in 7–14 days. Most patients fully recover."
    ),

    "Mild": TreatmentPlan(
        condition="Community-Acquired Pneumonia (mild)",
        severity="Mild",
        immediate_actions=[
            "Outpatient management suitable if CURB-65 = 0–1",
            "Confirm no oxygen requirement (SpO2 ≥ 95%)",
            "Prescribe oral antibiotics and send home",
        ],
        medications=[
            {"name": "Amoxicillin",   "dose": "500 mg oral TDS",         "duration": "5 days"},
            {"name": "Doxycycline",   "dose": "100 mg BD",               "duration": "5 days (penicillin allergy)"},
            {"name": "Clarithromycin","dose": "500 mg BD",               "duration": "5 days (atypical)"},
            {"name": "Paracetamol",   "dose": "500 mg–1 g every 6 hours","duration": "As needed for fever"},
        ],
        investigations=[
            "Chest X-ray (confirm diagnosis)",
            "Pulse oximetry",
            "FBC + CRP (if uncertain)",
            "Sputum culture (optional)",
        ],
        lifestyle=[
            "Rest at home — no strenuous activity",
            "Drink plenty of fluids",
            "Honey and lemon for cough relief",
            "Warm compress on chest for comfort",
            "Good hand hygiene to avoid spreading infection",
            "Complete the full antibiotic course",
        ],
        follow_up=[
            "Review in 48 hours if not improving",
            "Repeat CXR at 6 weeks to confirm resolution",
            "Vaccinate against flu and pneumococcus",
        ],
        red_flags=[
            "Fever not settling after 48 hours",
            "Coughing up blood",
            "Sudden worsening of breathing",
            "Chest pain on breathing",
        ],
        prognosis="Excellent prognosis with oral antibiotics. Full recovery expected within 7–10 days."
    ),
}


NORMAL_PLAN = TreatmentPlan(
    condition="No Pathology Detected",
    severity="Normal",
    immediate_actions=[
        "No immediate intervention required",
        "Correlate with clinical symptoms",
    ],
    medications=[
        {"name": "No medication required", "dose": "—", "duration": "—"},
    ],
    investigations=[
        "Clinical review if symptoms persist",
        "Spirometry if recurrent respiratory symptoms",
    ],
    lifestyle=[
        "Maintain healthy diet (Mediterranean-style recommended)",
        "Regular aerobic exercise (150 min/week)",
        "No smoking / vaping",
        "Annual flu vaccination recommended",
        "Maintain healthy BMI (18.5–25)",
    ],
    follow_up=[
        "Routine GP review as scheduled",
        "Annual health check if over 50",
    ],
    red_flags=[
        "Persistent cough > 3 weeks → further investigation",
        "Unexplained weight loss → refer for CT",
        "Haemoptysis → urgent referral",
    ],
    prognosis="X-ray appears normal. Maintain preventive health measures."
)


def get_treatment_plan(label: str, severity: str = "Moderate") -> TreatmentPlan:
    """
    Return the appropriate treatment plan.

    Args:
        label:    "Pneumonia" or "Normal"
        severity: "Severe", "Moderate", or "Mild"

    Returns:
        TreatmentPlan dataclass instance
    """
    if label == "Normal":
        return NORMAL_PLAN
    return PNEUMONIA_PROTOCOLS.get(severity, PNEUMONIA_PROTOCOLS["Moderate"])


def get_dominant_severity(findings: list) -> str:
    """Extract the highest severity level from a list of finding dicts."""
    order = {"Severe": 3, "Moderate": 2, "Mild": 1, "Normal": 0}
    if not findings:
        return "Normal"
    return max(findings, key=lambda f: order.get(f.get("severity", "Normal"), 0))["severity"]
