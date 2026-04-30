# 🩺 Medical Hybrid RAG + XGBoost Diagnostic Assistant (Backend)

A powerful AI-based medical diagnostic backend that combines:

* ⚡ **XGBoost** for disease prediction
* 📚 **RAG (Retrieval-Augmented Generation)** for medical knowledge grounding
* 🤖 **LLM (Gemini)** for natural language understanding and responses

This system processes user symptoms, predicts possible diseases, verifies them with medical knowledge, and generates intelligent explanations.

---

## 🚀 Features

* 🧠 Symptom extraction from natural language input
* 📊 Disease prediction using trained XGBoost model
* 🔍 RAG-based medical validation
* 💬 Follow-up Q&A using LLM
* ⚡ FastAPI backend with streaming responses
* 🔐 API key-based authentication
* 🌐 CORS-enabled for frontend integration

---

## 🏗️ Architecture Overview

```
User Input (Symptoms / Description)
        ↓
LLM → Extract Symptoms
        ↓
XGBoost → Predict Disease
        ↓
RAG → Retrieve Medical Knowledge
        ↓
LLM → Generate Explanation
        ↓
Response (Diagnosis + Reasoning)
```

---

## 📁 Project Structure

```
medmodel/
│
├── main.py                         # FastAPI entry point
├── requirements.txt               # Dependencies
├── .env                           # Environment variables
│
├── services/
│   ├── llm_service.py             # LLM interaction
│   ├── rag_service.py             # RAG retrieval logic
│   └── xgboost_service.py         # Prediction logic
│
├── schemas/
│   └── schemas.py                 # API request/response models
│
├── core/
│   └── constants.py               # Model configs
│
├── api/
│   └── dependencies.py            # Auth & dependencies
│
├── data/
│   ├── rag_disease_db.json        # Medical knowledge base
│   ├── symptom_vocab.json         # Symptom dictionary
│   ├── label_encoder.json         # Encoded labels
│   └── feature_dictionary.json    # Features mapping
│
├── training/
│   ├── train_model.py             # Model training script
│   └── xgboost_training_data.csv  # Dataset
│
└── pipeline/
    ├── medical_pipeline.py
    ├── medical_rag_xgboost_pipeline.py
    └── master.py
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/medical-rag-backend.git
cd medical-rag-backend
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in root:

```env
GEMINI_API_KEY=your_api_key_here
API_KEY=your_backend_auth_key
```

---

## ▶️ Running the Server

```bash
uvicorn main:app --reload
```

Server will run on:

```
http://127.0.0.1:8000
```

Swagger Docs:

```
http://127.0.0.1:8000/docs
```

---

## 📡 API Endpoints

### 1️⃣ Extract Symptoms

```
POST /analyze-description
```

* Input: Natural language symptoms
* Output: Structured symptom checkboxes

---

### 2️⃣ Generate Diagnosis

```
POST /generate-diagnosis
```

* Uses:

  * XGBoost → prediction
  * RAG → verification
  * LLM → explanation

---

### 3️⃣ Follow-up Questions

```
POST /followup
```

* Context-aware Q&A using RAG + LLM

---

### 4️⃣ Health Check

```
GET /health
```

---

## 🧠 Technologies Used

* **FastAPI** – Backend framework
* **XGBoost** – Machine learning model
* **RAG** – Knowledge retrieval system
* **Google Gemini API** – LLM
* **Python** – Core language

---

## 🔄 Pipeline Execution (Training)

To retrain the model:

```bash
python train_model.py
```

Or run full pipeline:

```bash
bash run_pipeline.sh
# OR
run_pipeline.bat
```

---

## 🔐 Security Notes

* Replace `"*"` in CORS before production
* Keep API keys secure
* Add rate limiting for production deployment

---

## 📌 Future Improvements

* User history storage (Supabase integration)
* Personalized diagnosis
* Multi-language support
* Better medical dataset expansion
* Deployment with Docker

---

## 👨‍💻 Author

**Anmol Kumar Jindal**
B.Tech CSE | AI + Software Development

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.
It is **not a substitute for professional medical advice**.

---

If you want, I can also:

* make a **GitHub description + tags**
* create a **professional project banner**
* or tailor this README for **resume impact 🔥**
