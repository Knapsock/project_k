import os
import time
import threading
import traceback
import re
import warnings

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# ---------------- SUPPRESS WARNINGS ---------------- #
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------- LOAD ENV ---------------- #
load_dotenv()

FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
FLASK_ENV = os.getenv("FLASK_ENV", "production")

if not FLASK_SECRET_KEY or not ADMIN_PASSWORD:
    raise RuntimeError(
        "Missing FLASK_SECRET_KEY or ADMIN_PASSWORD in .env file"
    )

# ---------------- FLASK APP ---------------- #
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

CORS(app)

app.secret_key = FLASK_SECRET_KEY

app.config.update({
    "UPLOAD_FOLDER": os.path.abspath("uploads"),
    "CHROMA_DB": os.path.abspath("chroma_db"),
    "ALLOWED_EXTENSIONS": {"pdf"},
    "MAX_CONTENT_LENGTH": 250 * 1024 * 1024,
    "ADMIN_PASSWORD": ADMIN_PASSWORD
})

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["CHROMA_DB"], exist_ok=True)

# ---------------- LLM + EMBEDDINGS ---------------- #
try:
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_HOST
    )

    llm = OllamaLLM(
        model="mistral",
        base_url=OLLAMA_HOST,
        temperature=0.2,
        num_predict=150
    )

    llm_components = {
        "embeddings": embeddings,
        "llm": llm
    }

except Exception as e:
    print("[ERROR] Ollama connection failed:", e)
    llm_components = None

# ---------------- MEMORY + CACHE ---------------- #
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_cache = {}

# ---------------- VECTORSTORE ---------------- #
main_vectorstore = None

main_store_dir = os.path.join(
    app.config["CHROMA_DB"],
    "main"
)

if os.path.exists(main_store_dir) and os.listdir(main_store_dir):
    try:
        main_vectorstore = Chroma(
            persist_directory=main_store_dir,
            embedding_function=llm_components["embeddings"]
        )

        print("[INFO] Main vectorstore loaded")

    except Exception as e:
        print("[WARN] Failed to load main vectorstore:", e)
        main_vectorstore = None

# ---------------- FILE VALIDATION ---------------- #
def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )

# ---------------- HOME ---------------- #
@app.route("/")
def index():
    return render_template("index.html")

# ---------------- ADMIN LOGIN ---------------- #
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():

    if request.method == "POST":

        if request.form.get("password") == app.config["ADMIN_PASSWORD"]:
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))

        return render_template(
            "admin_login.html",
            error="Invalid password"
        )

    return render_template("admin_login.html")

# ---------------- ADMIN LOGOUT ---------------- #
@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return redirect(url_for("index"))

# ---------------- ADMIN DASHBOARD ---------------- #
@app.route("/admin/dashboard")
def admin_dashboard():

    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    return render_template("admin_dashboard.html")

# ---------------- PROCESS PDF ---------------- #
def process_file(path, filename, uploaded, errors):

    global main_vectorstore

    try:
        loader = PyMuPDFLoader(path)

        raw_docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120
        )

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

# ---------------- PDF UPLOAD ---------------- #
@app.route("/admin/upload", methods=["POST"])
def upload_files():

    if not session.get("admin"):
        return jsonify({"error": "Unauthorized"}), 401

    if "pdfs" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("pdfs")

    uploaded = []
    errors = []
    threads = []

    for file in files:

        if not file or file.filename == "":
            continue

        if not allowed_file(file.filename):
            errors.append(f"Invalid file: {file.filename}")
            continue

        filename = secure_filename(file.filename)

        path = os.path.abspath(
            os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )
        )

        try:
            file.save(path)

        except Exception as e:
            print(f"[ERROR] Cannot save file {filename}: {e}")

            errors.append(
                f"{filename}: Save error: {e}"
            )

            continue

        thread = threading.Thread(
            target=process_file,
            args=(path, filename, uploaded, errors)
        )

        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return jsonify({
        "uploaded": uploaded,
        "errors": errors,
        "success": bool(uploaded)
    })

# ---------------- ASK QUESTION ---------------- #
@app.route("/ask", methods=["POST"])
def ask_question():

    try:
        # -------- CHECK LLM -------- #
        if llm_components is None:
            return jsonify({
                "error": "LLM not initialized"
            }), 500

        # -------- GET QUESTION -------- #
        data = request.get_json(force=True)

        question = data.get(
            "question",
            ""
        ).strip()

        if not question:
            return jsonify({
                "error": "Question cannot be empty"
            }), 400

        # -------- CHECK VECTORSTORE -------- #
        if main_vectorstore is None:
            return jsonify({
                "error": "No documents loaded"
            }), 400

        # -------- NORMALIZE QUESTION -------- #
        def normalize(q):

            q = q.lower().strip()

            q = re.sub(r'\bwhat is\b', '', q)

            q = re.sub(r'[^\w\s]', '', q)

            return q.strip()

        cache_key = normalize(question)

        # -------- CACHE -------- #
        if cache_key in qa_cache:

            return jsonify({
                "answer": qa_cache[cache_key],
                "cached": True
            })

               # -------- VECTOR SEARCH -------- #
        try:
            retriever = main_vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 6,
                    "fetch_k": 12
                }
            )

            docs = retriever.invoke(question)

            # Convert docs into scored format
            scored = [(doc, 1.0) for doc in docs]

        except Exception as vec_err:

            print("[ERROR] Vector search:", vec_err)

            return jsonify({
                "error": "Vector search failure"
            }), 500

        # -------- FILTER RELEVANT DOCS -------- #
        filtered_docs = []

        for doc, score in scored:

            print(f"[DEBUG] Similarity Score: {score}")

            # Lower score = better
            if score < 4:
                filtered_docs.append(doc)

        # -------- FALLBACK -------- #
        if not filtered_docs:

            fallback_prompt = f"""
Answer this question briefly and clearly using general knowledge.

Question:
{question}

Rules:
- Keep answer short
- Use simple language
- Explain naturally
"""

            fallback_response = llm_components["llm"].invoke(
                fallback_prompt
            ).strip()

            return jsonify({
                "answer": f"(General knowledge answer)\n\n{fallback_response}"
            })

        # -------- SOURCE LIST -------- #
        sources = list(set(
            doc.metadata.get("source", "Unknown")
            for doc in filtered_docs
        ))

        # -------- CONTEXT -------- #
        context = "\n---\n".join(
            doc.page_content[:800]
            for doc in filtered_docs
        )

        # -------- PROMPT -------- #
        prompt = f"""
You are an intelligent educational AI assistant.

Your job:
1. Analyze the provided PDF context carefully.
2. Answer the user's question in short, clear, human-friendly language.
3. Rephrase difficult concepts into simple understandable explanations.
4. If the exact answer is not found, try giving the closest relevant explanation from the context.
5. If nothing relevant exists, say:
"I could not find a relevant answer in the uploaded documents."

Rules:
- Keep answers concise.
- Use easy language.
- Avoid copying raw PDF text directly.
- Explain naturally like a teacher.
- Keep answer within 3-5 sentences maximum.

Context:
{context}

Question:
{question}

Answer:
"""

        # -------- LLM RESPONSE -------- #
        try:
            response = llm_components["llm"].invoke(
                prompt
            ).strip()

            response = re.sub(
                r'\n{3,}',
                '\n\n',
                response
            )

            response = response.strip()

        except Exception as llm_err:

            print("[ERROR] LLM invoke:", llm_err)

            return jsonify({
                "error": "Model response failed"
            }), 500

        # -------- CACHE STORE -------- #
        qa_cache[cache_key] = response

        # -------- RETURN RESPONSE -------- #
        return jsonify({
            "answer": response,
            "sources": sources,
            "cached": False
        })

    except Exception as e:

        print("[ERROR] /ask failed:", e)

        traceback.print_exc()

        return jsonify({
            "error": "Internal server error"
        }), 500

# ---------------- ERROR HANDLERS ---------------- #
@app.errorhandler(400)
def bad_request(error):

    return jsonify({
        "error": "Bad request",
        "message": str(error)
    }), 400

@app.errorhandler(404)
def not_found(error):

    return jsonify({
        "error": "Not found",
        "message": str(error)
    }), 404

@app.errorhandler(500)
def internal_error(error):

    return jsonify({
        "error": "Server error",
        "message": str(error)
    }), 500

# ---------------- RUN APP ---------------- #
if __name__ == "__main__":

    print("[INFO] FLASK_SECRET_KEY Loaded")
    print("[INFO] ADMIN_PASSWORD Loaded")
    print(f"[INFO] OLLAMA_HOST: {OLLAMA_HOST}")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=(FLASK_ENV == "development")
    )