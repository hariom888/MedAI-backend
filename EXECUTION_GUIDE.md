# Medical AI Pipeline — Execution Guide

Complete command reference for every script in the pipeline.

---

## Project Layout

```
project_root/
├── medical_pipeline.py              # Stage 1: DOCX → raw hybrid dataset
├── medical_rag_xgboost_pipeline.py  # Stage 2: RAG DB + XGBoost training CSV
├── train.py                         # Stage 3: XGBoost model training
├── run_pipeline.sh                  # Master orchestrator (runs all 3 stages)
├── requirements.txt                 # Python dependencies
│
├── meddocsp.docx                    # [INPUT] Your source medical document
├── docx_content.txt                 # Extracted text (auto-created or manual)
│
├── disease_templates.json           # [INPUT/OUTPUT] Disease definitions
├── symptom_vocab.json               # [INPUT/OUTPUT] Symptom vocabulary
├── feature_dictionary.json          # [INPUT/OUTPUT] Column metadata
├── label_encoder.json               # [INPUT/OUTPUT] Disease → integer map
├── class_distribution.json          # [INPUT/OUTPUT] Samples per disease
├── disease_stats.json               # [INPUT/OUTPUT] Per-disease stats
├── train.csv                        # [INPUT] Original training data
├── test.csv                         # [INPUT] Original test data
│
├── rag_disease_db.json              # [OUTPUT] Structured RAG database
├── xgboost_training_data.csv        # [OUTPUT] 250-sample/disease CSV
├── rag_chunks/                      # [OUTPUT] Markdown embedding chunks
│   └── *.md
│
└── output/                          # [OUTPUT] Model artifacts
    ├── medical_model.xgb
    ├── label_encoder.pkl
    ├── ordinal_encoder.pkl
    ├── classification_report.txt
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── shap_summary.png
    └── training_log.txt
```

---

## Step 0 — Install Dependencies

```bash
# Create and activate a virtual environment (strongly recommended)
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install all requirements
pip install -r requirements.txt

# Verify key packages installed correctly
python -c "import xgboost, shap, pandas, sklearn; print('All packages OK')"
```

---

## Option A — One-Command Full Pipeline (Recommended)

The shell script runs all three stages in order with error checking.

```bash
# Make executable (first time only)
chmod +x run_pipeline.sh

# Run the full pipeline (combined dataset, SHAP enabled)
./run_pipeline.sh

# ── Common variants ───────────────────────────────────────────────

# No source .docx file — skip Stage 1 (use pre-existing JSON files from the zip)
./run_pipeline.sh --skip-docx

# Skip Stage 1 AND Stage 2 — retrain the model only
./run_pipeline.sh --train-only

# Skip Stage 1 only (you have the JSONs but want to regenerate RAG + train)
./run_pipeline.sh --skip-docx

# Train on RAG data only (no original CSVs needed)
./run_pipeline.sh --mode rag --skip-docx

# Train on original pipeline data only
./run_pipeline.sh --mode original --skip-rag

# Add cross-validation + skip SHAP for speed
./run_pipeline.sh --cv --no-shap

# Enable GPU (requires CUDA-enabled XGBoost: pip install xgboost[gpu])
./run_pipeline.sh --gpu
```

---

## Option B — Run Each Script Individually

### Stage 1 — `medical_pipeline.py`

Parses your `.docx` source document and generates the full hybrid training dataset.

```bash
# Prerequisites: meddocsp.docx (or docx_content.txt) in project root

# Step 1a: Extract text from your DOCX (if not already done)
python -c "
from docx import Document
doc = Document('meddocsp.docx')
text = '\n===\n'.join([p.text for p in doc.paragraphs])
open('docx_content.txt', 'w').write(text)
print('Extracted to docx_content.txt')
"

# Step 1b: Run the pipeline (generates output/ directory with all CSVs + JSONs)
python medical_pipeline.py

# Expected output files:
#   output/train.csv                 ~27,000 rows
#   output/test.csv                  ~6,800 rows
#   output/disease_templates.json    238 disease definitions
#   output/symptom_vocab.json        695 symptom terms
#   output/feature_dictionary.json   column metadata
#   output/label_encoder.json        disease → integer map
#   output/class_distribution.json   samples per disease
#   output/disease_stats.json        per-disease stats
#   output/train_model.py            auto-generated training script
```

---

### Stage 2 — `medical_rag_xgboost_pipeline.py`

Builds the RAG database and generates the probabilistic XGBoost training CSV.

```bash
# Prerequisites:
#   disease_templates.json  (from Stage 1 or from the zip)
#   symptom_vocab.json
#   feature_dictionary.json
#   label_encoder.json
#   train.csv               (used to compute data-driven symptom prevalence)
#
# If you ran Stage 1, copy files to root:
cp output/disease_templates.json .
cp output/symptom_vocab.json .
cp output/feature_dictionary.json .
cp output/label_encoder.json .
cp output/class_distribution.json .
cp output/disease_stats.json .
cp output/train.csv .
cp output/test.csv .

# Run the RAG + XGBoost generator
python medical_rag_xgboost_pipeline.py

# Expected output:
#   rag_disease_db.json              597 KB structured RAG database
#   xgboost_training_data.csv        ~80 MB  59,500 rows × 696 columns
#   rag_chunks/*.md                  468 Markdown chunk files
```

---

### Stage 3 — `train.py`

Trains the XGBoost classifier. Supports multiple data-source modes and CLI flags.

```bash
# ── Basic usage ───────────────────────────────────────────────────

# Default: combined mode (merges original + RAG data)
python train.py

# Original pipeline data only (needs output/train.csv + output/test.csv)
python train.py --mode original \
  --original-train output/train.csv \
  --original-test  output/test.csv

# RAG data only (automatically splits 80/20)
python train.py --mode rag \
  --rag-csv xgboost_training_data.csv

# Combined mode (explicit paths)
python train.py --mode combined \
  --original-train output/train.csv \
  --original-test  output/test.csv \
  --rag-csv        xgboost_training_data.csv \
  --out            output

# ── Hyperparameter tuning ──────────────────────────────────────────

python train.py \
  --mode combined \
  --n-estimators 800 \
  --max-depth 10 \
  --lr 0.03 \
  --subsample 0.85 \
  --colsample 0.75 \
  --out output/tuned_run

# ── Cross-validation ───────────────────────────────────────────────

# Run 5-fold CV before final training (adds ~5-10 min on large data)
python train.py --mode combined --cv

# ── Speed optimizations ────────────────────────────────────────────

# Skip SHAP (saves ~2-3 min on large datasets)
python train.py --no-shap

# Fast dev run: RAG-only, no SHAP, no CV
python train.py --mode rag --no-shap --n-estimators 100

# ── GPU training (requires CUDA + GPU-enabled XGBoost) ─────────────

pip install "xgboost[gpu]"
# Then edit train.py and change in model_params:
#   tree_method = "hist"  →  tree_method = "gpu_hist"
#   device      = "cpu"   →  device      = "cuda"
python train.py --mode combined

# ── Custom output directory ────────────────────────────────────────

python train.py --mode combined --out results/experiment_01

# ── All flags together ─────────────────────────────────────────────

python train.py \
  --mode combined \
  --original-train output/train.csv \
  --original-test  output/test.csv \
  --rag-csv        xgboost_training_data.csv \
  --out            output \
  --n-estimators   600 \
  --max-depth      9 \
  --lr             0.04 \
  --subsample      0.85 \
  --colsample      0.80 \
  --cv \
  --seed           42
```

---

## Inference — Using the Trained Model

```python
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

# Load artifacts
model = xgb.XGBClassifier()
model.load_model("output/medical_model.xgb")

with open("output/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load ordinal encoder if the dataset had categorical columns
try:
    with open("output/ordinal_encoder.pkl", "rb") as f:
        oe = pickle.load(f)
    has_oe = True
except FileNotFoundError:
    has_oe = False

# Prepare a new patient row (all 695 symptom columns, 0 or 1)
import json
with open("feature_dictionary.json") as f:
    feat_dict = json.load(f)

symptom_cols = feat_dict["symptom_features"]

# Example: patient with fever, headache, neck stiffness
patient = {col: 0 for col in symptom_cols}
patient["fever"]          = 1
patient["headache"]       = 1
patient["neck_stiffness"] = 1
patient["confusion"]      = 1

X = pd.DataFrame([patient])[symptom_cols].fillna(0).astype("float32")

# Predict
y_pred  = model.predict(X)
y_proba = model.predict_proba(X)

disease  = le.inverse_transform(y_pred)[0]
conf     = y_proba[0].max() * 100

print(f"Predicted disease : {disease}")
print(f"Confidence        : {conf:.1f}%")

# Top-5 differential
top5_idx     = np.argsort(y_proba[0])[::-1][:5]
top5_labels  = le.classes_[top5_idx]
top5_probs   = y_proba[0][top5_idx] * 100
for lbl, prob in zip(top5_labels, top5_probs):
    print(f"  {prob:5.1f}%  {lbl}")
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: xgboost` | `pip install xgboost` |
| `ModuleNotFoundError: shap` | `pip install shap` or use `--no-shap` |
| `FileNotFoundError: train.csv` | Run Stage 1, or copy files from zip to project root |
| `ValueError: y contains new labels` in test | RAG CSV has diseases not in original; use `--mode rag` or `--mode combined` |
| Out of memory on large dataset | Reduce `--n-estimators`, add `--no-shap`, or use `--mode rag` |
| SHAP takes forever | Use `--no-shap` flag |
| Warnings about `use_label_encoder` | Safe to ignore; removed in XGBoost ≥ 2.0 |
| `bc: command not found` in shell script | Install `bc`: `sudo apt install bc` or `brew install bc` |
