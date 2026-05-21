# Project K: Local RAG Chatbot 🤖

An advanced, privacy-focused Retrieval-Augmented Generation (RAG) chatbot application built with Python. This system allows users to upload documents, parse text, embed data into a local vector database, and query a local Large Language Model (LLM) for intelligent, context-aware answers.

---

## 🌟 Key Features
* **Local LLM Integration:** Utilizes Ollama to run open-source models completely locally, ensuring complete data privacy.
* **Vector Database Storage:** Implements **ChromaDB** for fast, efficient vector similarity search and document retrieval.
* **Robust Backend API:** Powered by a lightweight **Flask** server handling secure request routing, document ingestion, and question-answering endpoints.
* **Containerization Ready:** Includes a `Dockerfile` and `docker-compose.yml` configurations for seamless deployment.

---

## 📁 Project Structure
```text
├── chroma_db/          # SQLite3-backed local vector store (ignored in git)
├── templates/          # Frontend UI HTML/CSS templates
├── uploads/            # Temporary storage for user-uploaded documents
├── .env                # Local environment secrets (API keys, ports)
├── .gitignore          # Keeps build caches and databases out of source control
├── app.py              # Main Flask application server and API endpoints
├── Dockerfile          # Container build specifications
├── docker-compose.yml  # Multi-container orchestration config
├── requirements.txt    # Project Python dependencies
└── README.md           # Project documentation
