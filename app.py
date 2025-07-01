import os
import shutil
import time
import threading
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Load environment variables
load_dotenv()
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
FLASK_ENV = os.getenv("FLASK_ENV", "production")

if not FLASK_SECRET_KEY or not ADMIN_PASSWORD:
    raise RuntimeError("Missing FLASK_SECRET_KEY or ADMIN_PASSWORD in .env file")

# Flask app setup
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = FLASK_SECRET_KEY
app.config.update({
    "UPLOAD_FOLDER": "uploads",
    "CHROMA_DB": "chroma_db",
    "ALLOWED_EXTENSIONS": {"pdf"},
    "MAX_CONTENT_LENGTH": 250 * 1024 * 1024,
    "ADMIN_PASSWORD": ADMIN_PASSWORD
})

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["CHROMA_DB"], exist_ok=True)

# LLM & Embedding setup
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST)
    llm = OllamaLLM(model="llama3", base_url=OLLAMA_HOST, temperature=0.3)
    llm_components = {"embeddings": embeddings, "llm": llm}
except Exception as e:
    print("[ERROR] Ollama connection failed:", e)
    llm_components = None

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_cache = {}

# Load main vectorstore
main_vectorstore = None
main_store_dir = os.path.join(app.config["CHROMA_DB"], "main")
if os.path.exists(main_store_dir):
    try:
        main_vectorstore = Chroma(persist_directory=main_store_dir, embedding_function=llm_components["embeddings"])
        print("[INFO] Main vectorstore loaded")
    except Exception as e:
        print("[WARN] Failed to load main vectorstore:", e)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if request.form.get("password") == app.config["ADMIN_PASSWORD"]:
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))
        return render_template("admin_login.html", error="Invalid password")
    return render_template("admin_login.html")

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return redirect(url_for("index"))

@app.route("/admin/dashboard")
def admin_dashboard():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    return render_template("admin_dashboard.html")

def process_file(path, filename, uploaded, errors):
    global main_vectorstore
    try:
        loader = PyMuPDFLoader(path)
        raw_docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        docs = splitter.split_documents(raw_docs)

        for doc in docs:
            doc.metadata["source"] = filename

        if main_vectorstore is None:
            main_vectorstore = Chroma.from_documents(
                docs,
                embedding=llm_components["embeddings"],
                persist_directory=main_store_dir
            )
        else:
            main_vectorstore.add_documents(docs)
        main_vectorstore.persist()
        uploaded.append(filename)
    except Exception as e:
        errors.append(f"{filename}: {e}")
    finally:
        for _ in range(3):
            try:
                if os.path.exists(path):
                    os.remove(path)
                break
            except PermissionError:
                time.sleep(1)

@app.route("/admin/upload", methods=["POST"])
def upload_files():
    if not session.get("admin"):
        return jsonify({"error": "Unauthorized"}), 401
    if "pdfs" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("pdfs")
    uploaded, errors = [], []
    threads = []

    for file in files:
        if not file or file.filename == "":
            continue
        if not allowed_file(file.filename):
            errors.append(f"Invalid file: {file.filename}")
            continue

        filename = secure_filename(file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        thread = threading.Thread(target=process_file, args=(path, filename, uploaded, errors))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return jsonify({"uploaded": uploaded, "errors": errors, "success": bool(uploaded)})

@app.route("/ask", methods=["POST"])
def ask_question():
    if llm_components is None:
        return jsonify({"error": "LLM not initialized"}), 500

    data = request.get_json()
    question = data.get("question", "").strip().lower()

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400
    if main_vectorstore is None:
        return jsonify({"error": "No documents loaded. Please ask admin to upload files."}), 400

    if question in qa_cache:
        return jsonify({"answer": qa_cache[question], "cached": True})

    try:
        scored = main_vectorstore.similarity_search_with_score(question, k=1)
        relevant_docs = sorted(scored, key=lambda x: x[1])[:1]
        relevant_docs = [doc for doc, _ in relevant_docs]

        context = "\n---\n".join(doc.page_content[:300] for doc in relevant_docs)

        prompt = f"""You are a highly knowledgeable technical assistant. Your task is to answer the user's question as clearly, deeply, and informatively as possible, using only the provided context.

Context:
{context}

Question:
{question}

Guidelines:
- Provide a detailed explanation
- Use bullet points or numbered steps
- Include definitions or examples
- Avoid vague answers
- If the answer is not in the context, say:
  \"The documents do not contain sufficient information to answer this question.\"

Final Answer:"""

        response = llm_components["llm"].invoke(prompt).strip()
        qa_cache[question] = response

        return jsonify({"answer": response, "cached": False})

    except Exception as e:
        print("[ERROR] during ask:", e)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    print("[INFO] FLASK_SECRET_KEY Loaded")
    print("[INFO] ADMIN_PASSWORD Loaded")
    print(f"[INFO] OLLAMA_HOST: {OLLAMA_HOST}")
    app.run(host="0.0.0.0", port=5000, debug=(FLASK_ENV == "development"))
