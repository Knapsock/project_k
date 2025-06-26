from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import os
import time
import shutil

app = Flask(__name__)
vectorstore = None

# Ensure upload and database directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    global vectorstore
    
    # Clear previous uploads and database
    if os.path.exists("uploads"):
        shutil.rmtree("uploads")
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)
    
    if 'pdfs' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    files = request.files.getlist("pdfs")
    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No selected files"}), 400

    uploaded = []
    errors = []
    all_docs = []

    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            errors.append(f"File {file.filename} is not a PDF")
            continue

        try:
            filename = file.filename
            filepath = os.path.join("uploads", filename)
            file.save(filepath)
            
            # Load and split PDF
            loader = PyMuPDFLoader(filepath)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            docs = splitter.split_documents(documents)
            all_docs.extend(docs)
            uploaded.append(filename)
        except Exception as e:
            errors.append(f"Error processing {file.filename}: {str(e)}")

    if all_docs:
        try:
            # Create combined vectorstore
            persist_dir = os.path.join("chroma_db", "combined")
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectorstore = Chroma.from_documents(
                documents=all_docs,
                embedding=embeddings,
                persist_directory=persist_dir
            )
            vectorstore.persist()
        except Exception as e:
            errors.append(f"Error creating vectorstore: {str(e)}")

    response = {
        "message": f"Successfully processed {len(uploaded)} files" if uploaded else "No files processed",
        "uploaded": uploaded,
        "errors": errors
    }
    return jsonify(response)

@app.route("/ask", methods=["POST"])
def ask_question():
    global vectorstore
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400
    if not vectorstore:
        return jsonify({"error": "Please upload PDFs first"}), 400

    try:
        start_time = time.time()
        
        # Retrieve relevant documents
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Generate prompt
        prompt = f"""You are a helpful maritime expert assistant. Answer the question based only on the following context:
        
        Context:
        {context}

        Question: {question}

        Answer in clear, concise terms. If the answer isn't in the context, say you don't know."""

        # Get answer from LLM
        llm = OllamaLLM(model="llama3")
        answer = llm.invoke(prompt)
        
        return jsonify({
            "answer": answer.strip(),
            "time": f"{time.time() - start_time:.2f}s",
            "sources": [os.path.basename(doc.metadata.get('source', 'unknown')) for doc in docs]
        })
    except Exception as e:
        return jsonify({"error": f"Error processing question: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)