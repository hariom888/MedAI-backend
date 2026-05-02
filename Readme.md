# 🏥 MedAI Backend – AI Powered Healthcare Assistant

> A scalable **RAG-based AI healthcare backend** that provides intelligent medical assistance using LLMs, vector databases, and real-time APIs.

---

## 🔗 Project Links

* 🌐 **Frontend Repository:**
  👉 [https://github.com/Anmol-pi/MedAI-Frontend](https://github.com/Anmol-pi/MedAI-Frontend)

* 🚀 **Backend Deployment (Live API):**
  👉 [https://medai-backend-production-a830.up.railway.app](https://medai-backend-production-a830.up.railway.app)

---

## 📌 Overview

MedAI is an AI-powered healthcare assistant designed to deliver **accurate, context-aware medical responses** using a **Retrieval-Augmented Generation (RAG)** pipeline.

The backend integrates advanced AI models with a vector database to ensure responses are **relevant, factual, and domain-specific**, making it suitable for real-world healthcare support systems.

---

## ⚙️ Key Features

* 🧠 **RAG-based AI System**
  Combines LLMs with vector search for accurate answers

* 🔍 **Semantic Search with Vector DB**
  Uses embeddings to retrieve relevant medical context

* 🤖 **LLM Integration**
  Generates human-like, contextual responses

* 📂 **Document Processing Pipeline**
  Supports ingestion of medical datasets / knowledge sources

* ⚡ **FastAPI Backend**
  High-performance API endpoints for frontend integration

* 🔐 **Scalable Architecture**
  Designed for production deployment (Railway)

---

## 🏗️ Tech Stack

### 🚀 Backend

* Python
* FastAPI

### 🧠 AI / ML

* LLM APIs (Groq / LLaMA / etc.)
* Embeddings

### 🗄️ Database

* Vector Database (Qdrant)

### 🔗 Integration

* REST APIs
* Frontend (React / TypeScript)

---

## 📊 System Architecture

```text
User Query
   ↓
Frontend (React)
   ↓
FastAPI Backend
   ↓
Embedding Generation
   ↓
Vector Search (Qdrant)
   ↓
Context Retrieval
   ↓
LLM (Groq / LLaMA)
   ↓
Final AI Response
```

---

## 📁 Project Structure

```text
MedAI-backend/
│── app/
│   ├── routes/          # API endpoints
│   ├── services/        # Business logic
│   ├── rag/             # RAG pipeline
│   ├── db/              # Database connections
│   └── utils/           # Helper functions
│
│── data/                # Documents / embeddings
│── main.py              # Entry point
│── requirements.txt     # Dependencies
│── .env                 # Environment variables
```

---

## 🚀 Getting Started

### 🔧 Prerequisites

* Python 3.9+
* API Keys (Groq / Cohere / etc.)
* Qdrant setup

---

### 📥 Installation

```bash
git clone <your-repo-url>
cd MedAI-backend
pip install -r requirements.txt
```

---

### 🔑 Environment Setup

Create a `.env` file:

```env
GROQ_API_KEY=your_key
QDRANT_URL=your_url
QDRANT_API_KEY=your_key
```

---

### ▶️ Run the Server

```bash
uvicorn main:app --reload
```

---

## 📡 API Endpoints (Example)

| Method | Endpoint | Description           |
| ------ | -------- | --------------------- |
| GET    | `/`      | Health check          |
| POST   | `/query` | Ask medical questions |

---

## 🧠 How It Works

The system follows a **Retrieval-Augmented Generation pipeline**:

1. User sends a query
2. Query is converted into embeddings
3. Relevant documents are retrieved from vector DB
4. Context is passed to LLM
5. LLM generates final response

---

## 🌟 Use Cases

* AI Medical Chatbot
* Health Assistance Platforms
* Clinical Decision Support (basic level)
* Educational Medical Tools

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and should not be used as a substitute for professional medical advice.

---

## 👨‍💻 Contributors

* **Anmol Kumar Jindal**
* **HariOm**

---

## 🤝 Contributing

Contributions are welcome!

```bash
fork → clone → create branch → commit → push → PR
```

---

## 📄 License

This project is licensed under the **MIT License**

---

## ⭐ Support

If you like this project:

* ⭐ Star the repo
* 🍴 Fork it
* 🧠 Share ideas
