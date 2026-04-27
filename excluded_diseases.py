"""Shared disease exclusion rules for dataset rebuilds and training."""

from __future__ import annotations

EXCLUDED_DISEASES = {
    "Acne",
    "Acute Cholecystitis",
    "Acute Lymphoblastic Leukaemia",
    "Acute Myeloid Leukaemia",
    "Achilles Tendinopathy",
    "Abdominal Aortic Aneurysm",
    "Acute Respiratory Infection",
    "Adenomyosis",
    "Alopecia",
    "Allergic Rhinitis",
    "Allergies",
    "Anal Cancer",
    "Anaphylaxis",
    "Anemia",
    "Ankle Avulsion Fracture",
    "Anorexia Nervosa",
    "Arthritis",
    "Ankle Sprain",
    "Arterial Thrombosis",
    "Atopic Eczema",
    "Attention Deficit Hyperactivity Disorder",
    "Atrial Fibrillation",
    "Bronchitis",
    "Boils",
    "Cardiomyopathy",
    "Central Nervous System Tumor",
    "Chronic Kidney Disease",
    "Chronic Obstructive Pulmonary Disease",
    "Cirrhosis",
    "Colorectal Cancer",
    "Coronary Artery Disease",
    "COVID-19",
    "Crohn's Disease",
    "Cryptosporidiosis",
    "Deep Vein Thrombosis",
    "Dengue",
    "Depression",
    "Dermatitis Herpetiformis",
    "Diarrhea",
    "Drug Use and Addiction",
    "Eating Disorders",
    "Eczema",
    "Endocarditis",
    "Fatty Liver Disease",
    "Filariasis",
    "Gastroesophageal Reflux Disease",
    "Gastritis",
    "Giardiasis",
    "Gout",
    "H1N1 Swine Flu",
    "Heart Failure",
    "Hepatitis C",
    "Hepatitis E",
    "High Cholesterol",
    "HIV/AIDS",
    "Hives",
    "Infertility",
    "Jaundice",
    "Leishmaniasis",
    "Leptospirosis",
    "Lung Cancer",
    "Malaria",
    "Measles",
    "Melasma",
    "Metabolic Syndrome",
    "Middle East Respiratory Syndrome",
    "Myocarditis",
    "Norovirus Infection",
    "Obesity",
    "Occupational Lung Disease",
    "Osteomyelitis",
    "Osteoporosis",
    "Pancreatitis",
    "Peptic Ulcer Disease",
    "Pericarditis",
    "Peripheral Artery Disease",
    "Pleural Effusion",
    "Polycystic Ovary Syndrome",
    "Psoriasis",
    "Pulmonary Fibrosis",
    "Rashes",
    "Rheumatic Heart Disease",
    "Rotavirus Infection",
    "Salmonella Infection",
    "Sarcoidosis",
    "Scabies",
    "Scoliosis",
    "Severe Acute Respiratory Syndrome",
    "Shigellosis",
    "Skin Cancer",
    "Sleep Apnea",
    "Ulcerative Colitis",
    "Urinary Tract Infection",
    "Vitiligo",
}

ALIASES_TO_CANONICAL = {
    "CAD": "Coronary Artery Disease",
    "COPD": "Chronic Obstructive Pulmonary Disease",
    "CNS Tumor": "Central Nervous System Tumor",
    "Central Nervous System Tumor": "Central Nervous System Tumor",
    "Coronary Artery Disease": "Coronary Artery Disease",
    "PCOS": "Polycystic Ovary Syndrome",
    "Polycystic Ovary Syndrome": "Polycystic Ovary Syndrome",
    "MERS": "Middle East Respiratory Syndrome",
    "Middle East Respiratory Syndrome": "Middle East Respiratory Syndrome",
    "SARS": "Severe Acute Respiratory Syndrome",
    "Severe Acute Respiratory Syndrome": "Severe Acute Respiratory Syndrome",
    "H1N1": "H1N1 Swine Flu",
    "H1N1 Swine Flu": "H1N1 Swine Flu",
}


def canonicalize_disease_name(name: str) -> str:
    """Map shorthand disease names to the canonical labels used in the data."""
    return ALIASES_TO_CANONICAL.get(name, name)


def is_excluded_disease(name: str) -> bool:
    """Return True when the disease should be removed from training assets."""
    return canonicalize_disease_name(name) in EXCLUDED_DISEASES


def filter_excluded_disease_names(names):
    """Keep only disease names that are not excluded."""
    return [name for name in names if not is_excluded_disease(name)]


def filter_excluded_disease_rows(rows, key: str = "label"):
    """Keep only rows whose disease field is not excluded."""
    return [row for row in rows if not is_excluded_disease(row.get(key, ""))]
