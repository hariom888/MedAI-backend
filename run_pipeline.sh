#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_pipeline.sh  —  Full Medical AI Pipeline Orchestrator
# ══════════════════════════════════════════════════════════════════════════════
#
# USAGE:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh                        # full pipeline (all 3 stages)
#   ./run_pipeline.sh --skip-docx            # skip medical_pipeline.py (no .docx)
#   ./run_pipeline.sh --skip-rag             # skip RAG pipeline
#   ./run_pipeline.sh --train-only           # only run train.py
#   ./run_pipeline.sh --mode rag             # train on RAG data only
#   ./run_pipeline.sh --mode original        # train on original data only
#   ./run_pipeline.sh --mode combined        # train on merged data (default)
#   ./run_pipeline.sh --cv                   # add cross-validation step
#   ./run_pipeline.sh --no-shap              # skip SHAP (faster)
#   ./run_pipeline.sh --gpu                  # enable GPU training
#
# PREREQUISITES:
#   pip install -r requirements.txt
#   Place your meddocsp.docx (or extracted docx_content.txt) in the project root.
#
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail     # exit on any error, undefined var, or pipe failure

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
section() { echo -e "\n${BOLD}══════════════════════════════════════════${NC}"; \
            echo -e "${BOLD}  $*${NC}"; \
            echo -e "${BOLD}══════════════════════════════════════════${NC}"; }

# ── Argument defaults ─────────────────────────────────────────────────────────
SKIP_DOCX=false
SKIP_RAG=false
TRAIN_ONLY=false
TRAIN_MODE="combined"
RUN_CV=""
NO_SHAP=""
GPU=false
PYTHON="${PYTHON:-python3}"          # override with PYTHON=python ./run_pipeline.sh

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-docx)   SKIP_DOCX=true  ;;
    --skip-rag)    SKIP_RAG=true   ;;
    --train-only)  TRAIN_ONLY=true ;;
    --mode)        TRAIN_MODE="$2"; shift ;;
    --cv)          RUN_CV="--cv"   ;;
    --no-shap)     NO_SHAP="--no-shap" ;;
    --gpu)         GPU=true        ;;
    -h|--help)
      grep '^#' "$0" | grep -v '#!/' | sed 's/^# \{0,2\}//'
      exit 0 ;;
    *) error "Unknown argument: $1. Use --help for usage." ;;
  esac
  shift
done

# ── Environment check ─────────────────────────────────────────────────────────
section "Environment Check"

command -v "$PYTHON" >/dev/null 2>&1 || error "Python not found. Set PYTHON= or install Python 3.10+."
PYVER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python version : $PYVER"
[[ $(echo "$PYVER >= 3.10" | bc -l) -eq 1 ]] || error "Python 3.10+ required (found $PYVER)"

# Check key packages
for PKG in pandas numpy sklearn xgboost matplotlib seaborn; do
  "$PYTHON" -c "import $PKG" 2>/dev/null && success "  $PKG installed" \
    || warn "  $PKG NOT found — run: pip install -r requirements.txt"
done

# Check for shap only if not skipped
if [[ -z "$NO_SHAP" ]]; then
  "$PYTHON" -c "import shap" 2>/dev/null && success "  shap installed" \
    || warn "  shap NOT found — use --no-shap flag or: pip install shap"
fi

# Output directories
mkdir -p output rag_chunks
info "Output dirs    : output/  rag_chunks/"

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — medical_pipeline.py
# Parses the .docx source document and generates the original hybrid dataset.
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$TRAIN_ONLY" == "false" && "$SKIP_DOCX" == "false" ]]; then
  section "Stage 1 — medical_pipeline.py (Document → Raw Dataset)"

  # Check for the source document
  if [[ ! -f "docx_content.txt" && ! -f "meddocsp.docx" ]]; then
    warn "Neither docx_content.txt nor meddocsp.docx found."
    warn "Skipping Stage 1. If you have the source document, extract its text to docx_content.txt."
    warn "  python -c \"from docx import Document; d=Document('meddocsp.docx'); open('docx_content.txt','w').write('\n===\n'.join([p.text for p in d.paragraphs]))\""
    SKIP_DOCX=true
  fi

  if [[ "$SKIP_DOCX" == "false" ]]; then
    info "Running medical_pipeline.py…"
    "$PYTHON" medical_pipeline.py

    # Verify outputs
    for F in output/train.csv output/test.csv output/disease_templates.json \
              output/symptom_vocab.json output/feature_dictionary.json \
              output/label_encoder.json output/class_distribution.json; do
      [[ -f "$F" ]] && success "  Generated: $F" || warn "  Missing: $F"
    done
  fi
else
  info "Stage 1 skipped (--skip-docx or --train-only)."
fi

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — medical_rag_xgboost_pipeline.py
# Reads the JSON artefacts and generates:
#   • rag_disease_db.json         — structured RAG database (238 diseases)
#   • rag_chunks/*.md             — Markdown chunks for vector embedding
#   • xgboost_training_data.csv  — 250 samples × disease tabular dataset
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$TRAIN_ONLY" == "false" && "$SKIP_RAG" == "false" ]]; then
  section "Stage 2 — medical_rag_xgboost_pipeline.py (RAG + XGBoost Data)"

  # This script needs disease_templates.json and symptom_vocab.json.
  # Prefer output/ versions if Stage 1 ran; fall back to root-level originals.
  if [[ ! -f "disease_templates.json" && -f "output/disease_templates.json" ]]; then
    info "Symlinking output/ JSON files to project root for pipeline…"
    for F in disease_templates.json symptom_vocab.json feature_dictionary.json \
              label_encoder.json class_distribution.json disease_stats.json; do
      [[ -f "output/$F" && ! -f "$F" ]] && ln -sf "output/$F" "$F" && info "  Linked: $F"
    done
  fi

  # Verify required input files
  MISSING=false
  for F in disease_templates.json symptom_vocab.json feature_dictionary.json \
            label_encoder.json train.csv; do
    [[ -f "$F" || -f "output/$F" ]] || { warn "Required input not found: $F"; MISSING=true; }
  done

  if [[ "$MISSING" == "true" ]]; then
    warn "Some input files missing — Stage 2 may fail. Ensure Stage 1 ran first or"
    warn "place the JSON files from the zip in the project root."
  fi

  info "Running medical_rag_xgboost_pipeline.py…"
  "$PYTHON" medical_rag_xgboost_pipeline.py

  [[ -f "rag_disease_db.json" ]]        && success "  Generated: rag_disease_db.json"
  [[ -f "xgboost_training_data.csv" ]]  && success "  Generated: xgboost_training_data.csv"
  RAG_CHUNKS=$(ls rag_chunks/*.md 2>/dev/null | wc -l)
  success "  Generated: $RAG_CHUNKS RAG chunk files in rag_chunks/"

else
  info "Stage 2 skipped (--skip-rag or --train-only)."
fi

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — train.py
# Trains the XGBoost model on the combined (or specified) dataset.
# ══════════════════════════════════════════════════════════════════════════════

section "Stage 3 — train.py (XGBoost Model Training)"

# Build train.py command
TRAIN_CMD="$PYTHON train.py"
TRAIN_CMD+=" --mode $TRAIN_MODE"
TRAIN_CMD+=" --out output"
TRAIN_CMD+=" --n-estimators 500"
TRAIN_CMD+=" --max-depth 8"
TRAIN_CMD+=" --lr 0.05"
TRAIN_CMD+=" --subsample 0.85"
TRAIN_CMD+=" --colsample 0.80"
[[ -n "$RUN_CV"  ]] && TRAIN_CMD+=" $RUN_CV"
[[ -n "$NO_SHAP" ]] && TRAIN_CMD+=" $NO_SHAP"

# GPU mode: override tree method via env (train.py reads from model_params)
if [[ "$GPU" == "true" ]]; then
  warn "GPU mode enabled — ensure CUDA XGBoost is installed."
  export XGB_DEVICE="cuda"
fi

info "Command: $TRAIN_CMD"
echo ""
eval "$TRAIN_CMD"

# ── Final summary ─────────────────────────────────────────────────────────────
section "Pipeline Complete"

echo ""
echo -e "  ${GREEN}Artifacts:${NC}"
for F in output/medical_model.xgb output/label_encoder.pkl \
          output/ordinal_encoder.pkl output/classification_report.txt \
          output/confusion_matrix.png output/feature_importance.png \
          output/shap_summary.png output/training_log.txt; do
  [[ -f "$F" ]] && echo -e "    ${GREEN}✓${NC}  $F" || echo -e "    ${YELLOW}–${NC}  $F (not generated)"
done

echo ""
echo -e "  ${CYAN}RAG artifacts:${NC}"
[[ -f "rag_disease_db.json" ]]       && echo -e "    ${GREEN}✓${NC}  rag_disease_db.json"
RAG_CHUNKS=$(ls rag_chunks/*.md 2>/dev/null | wc -l)
[[ "$RAG_CHUNKS" -gt 0 ]]            && echo -e "    ${GREEN}✓${NC}  rag_chunks/ ($RAG_CHUNKS .md files)"
[[ -f "xgboost_training_data.csv" ]] && echo -e "    ${GREEN}✓${NC}  xgboost_training_data.csv"
echo ""
