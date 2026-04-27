param(
    [switch]$SkipDocx,
    [switch]$SkipRag,
    [switch]$TrainOnly,
    [ValidateSet("original", "rag", "combined")]
    [string]$Mode = "combined",
    [switch]$Cv,
    [switch]$NoShap,
    [switch]$Gpu,
    [string]$Python = $(if ($env:PYTHON) { $env:PYTHON } else { "python" }),
    [switch]$Help
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

function Write-Info($Message) {
    Write-Host "[INFO]  $Message" -ForegroundColor Cyan
}

function Write-Success($Message) {
    Write-Host "[OK]    $Message" -ForegroundColor Green
}

function Write-WarnMsg($Message) {
    Write-Host "[WARN]  $Message" -ForegroundColor Yellow
}

function Fail($Message) {
    Write-Host "[ERROR] $Message" -ForegroundColor Red
    exit 1
}

function Write-Section($Title) {
    $line = "=" * 42
    Write-Host ""
    Write-Host $line -ForegroundColor White
    Write-Host "  $Title" -ForegroundColor White
    Write-Host $line -ForegroundColor White
}

function Show-Help {
    @"
run_pipeline.ps1 - Full Medical AI Pipeline Orchestrator

Usage:
  powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
  powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -SkipDocx
  powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -SkipRag
  powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -TrainOnly
  powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -Mode rag
  powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -Mode original
  powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -Mode combined
  powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -Cv
  powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -NoShap
  powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -Gpu
  powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -Python .\.venv\Scripts\python.exe

Parameters:
  -SkipDocx   Skip medical_pipeline.py
  -SkipRag    Skip medical_rag_xgboost_pipeline.py
  -TrainOnly  Only run train.py
  -Mode       original | rag | combined
  -Cv         Add cross-validation step
  -NoShap     Skip SHAP
  -Gpu        Set XGB_DEVICE=cuda
  -Python     Python executable path
  -Help       Show this help
"@ | Write-Host
}

if ($Help) {
    Show-Help
    exit 0
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Test-PythonImport($ModuleName) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Python
    $psi.Arguments = "-c `"import $ModuleName`""
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    [void]$proc.Start()
    [void]$proc.StandardOutput.ReadToEnd()
    [void]$proc.StandardError.ReadToEnd()
    $proc.WaitForExit()
    return $proc.ExitCode -eq 0
}

function Copy-IfMissing($Source, $Destination) {
    if ((Test-Path $Source) -and -not (Test-Path $Destination)) {
        Copy-Item -Path $Source -Destination $Destination
        Write-Info "  Copied: $Destination"
    }
}

Write-Section "Environment Check"

try {
    $null = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
} catch {
    Fail "Python not found. Set -Python or install Python 3.10+."
}

$pyVer = (& $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
Write-Info "Python version : $pyVer"

$versionObj = [version]$pyVer
if ($versionObj -lt [version]"3.10") {
    Fail "Python 3.10+ required (found $pyVer)"
}

foreach ($pkg in @("pandas", "numpy", "sklearn", "xgboost", "matplotlib", "seaborn")) {
    if (Test-PythonImport $pkg) {
        Write-Success "  $pkg installed"
    } else {
        Write-WarnMsg "  $pkg NOT found - run: pip install -r requirements.txt"
    }
}

if (-not $NoShap) {
    if (Test-PythonImport "shap") {
        Write-Success "  shap installed"
    } else {
        Write-WarnMsg "  shap NOT found - use -NoShap or run: pip install shap"
    }
}

New-Item -ItemType Directory -Force -Path "output" | Out-Null
New-Item -ItemType Directory -Force -Path "rag_chunks" | Out-Null
Write-Info "Output dirs    : output\  rag_chunks\"

if (-not $TrainOnly -and -not $SkipDocx) {
    Write-Section "Stage 1 - medical_pipeline.py (Document to Raw Dataset)"

    if (-not (Test-Path "docx_content.txt") -and -not (Test-Path "meddocsp.docx")) {
        Write-WarnMsg "Neither docx_content.txt nor meddocsp.docx found."
        Write-WarnMsg "Skipping Stage 1. If you have the source document, extract its text to docx_content.txt."
        Write-WarnMsg "  python -c `"from docx import Document; d=Document('meddocsp.docx'); open('docx_content.txt','w').write('\n===\n'.join([p.text for p in d.paragraphs]))`""
        $SkipDocx = $true
    }

    if (-not $SkipDocx) {
        Write-Info "Running medical_pipeline.py..."
        & $Python "medical_pipeline.py"
        if ($LASTEXITCODE -ne 0) {
            Fail "medical_pipeline.py failed."
        }

        foreach ($file in @(
            "output/train.csv",
            "output/test.csv",
            "output/disease_templates.json",
            "output/symptom_vocab.json",
            "output/feature_dictionary.json",
            "output/label_encoder.json",
            "output/class_distribution.json"
        )) {
            if (Test-Path $file) {
                Write-Success "  Generated: $file"
            } else {
                Write-WarnMsg "  Missing: $file"
            }
        }
    }
} else {
    Write-Info "Stage 1 skipped (-SkipDocx or -TrainOnly)."
}

if (-not $TrainOnly -and -not $SkipRag) {
    Write-Section "Stage 2 - medical_rag_xgboost_pipeline.py (RAG + XGBoost Data)"

    Write-Info "Ensuring root-level pipeline inputs are available..."
    foreach ($file in @(
        "disease_templates.json",
        "symptom_vocab.json",
        "feature_dictionary.json",
        "label_encoder.json",
        "class_distribution.json",
        "disease_stats.json",
        "train.csv"
    )) {
        Copy-IfMissing "output\$file" $file
    }

    $missing = $false
    foreach ($file in @(
        "disease_templates.json",
        "symptom_vocab.json",
        "feature_dictionary.json",
        "label_encoder.json",
        "train.csv"
    )) {
        if (-not (Test-Path $file) -and -not (Test-Path "output\$file")) {
            Write-WarnMsg "Required input not found: $file"
            $missing = $true
        }
    }

    if ($missing) {
        Write-WarnMsg "Some input files are missing - Stage 2 may fail."
    }

    Write-Info "Running medical_rag_xgboost_pipeline.py..."
    & $Python "medical_rag_xgboost_pipeline.py"
    if ($LASTEXITCODE -ne 0) {
        Fail "medical_rag_xgboost_pipeline.py failed."
    }

    if (Test-Path "rag_disease_db.json") {
        Write-Success "  Generated: rag_disease_db.json"
    }
    if (Test-Path "xgboost_training_data.csv") {
        Write-Success "  Generated: xgboost_training_data.csv"
    }
    $ragChunks = @(Get-ChildItem -Path "rag_chunks" -Filter "*.md" -ErrorAction SilentlyContinue).Count
    Write-Success "  Generated: $ragChunks RAG chunk files in rag_chunks\"
} else {
    Write-Info "Stage 2 skipped (-SkipRag or -TrainOnly)."
}

Write-Section "Stage 3 - train.py (XGBoost Model Training)"

$trainArgs = @(
    "train.py",
    "--mode", $Mode,
    "--out", "output",
    "--n-estimators", "500",
    "--max-depth", "8",
    "--lr", "0.05",
    "--subsample", "0.85",
    "--colsample", "0.80"
)

if ($Cv) {
    $trainArgs += "--cv"
}
if ($NoShap) {
    $trainArgs += "--no-shap"
}

if ($Gpu) {
    Write-WarnMsg "GPU mode enabled - ensure CUDA XGBoost is installed."
    $env:XGB_DEVICE = "cuda"
}

Write-Info ("Command: {0} {1}" -f $Python, ($trainArgs -join " "))
Write-Host ""

& $Python @trainArgs
if ($LASTEXITCODE -ne 0) {
    Fail "train.py failed."
}

Write-Section "Pipeline Complete"

Write-Host ""
Write-Host "  Artifacts:" -ForegroundColor Green
foreach ($file in @(
    "output/medical_model.xgb",
    "output/label_encoder.pkl",
    "output/ordinal_encoder.pkl",
    "output/classification_report.txt",
    "output/confusion_matrix.png",
    "output/feature_importance.png",
    "output/shap_summary.png",
    "output/training_log.txt"
)) {
    if (Test-Path $file) {
        Write-Host "    +  $file" -ForegroundColor Green
    } else {
        Write-Host "    -  $file (not generated)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "  RAG artifacts:" -ForegroundColor Cyan
if (Test-Path "rag_disease_db.json") {
    Write-Host "    +  rag_disease_db.json" -ForegroundColor Green
}
$finalRagChunks = @(Get-ChildItem -Path "rag_chunks" -Filter "*.md" -ErrorAction SilentlyContinue).Count
if ($finalRagChunks -gt 0) {
    Write-Host "    +  rag_chunks\ ($finalRagChunks .md files)" -ForegroundColor Green
}
if (Test-Path "xgboost_training_data.csv") {
    Write-Host "    +  xgboost_training_data.csv" -ForegroundColor Green
}
Write-Host ""
