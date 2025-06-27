from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import os
import time

app = Flask(__name__)
vectorstores = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    files = request.files.getlist("pdfs")
    uploaded = []
    errors = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            continue

        filename = file.filename
        filepath = os.path.join("uploads", filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)

        try:
            loader = PyMuPDFLoader(filepath)
            documents = loader.load()
            if not documents:
                errors.append(f"❌ No text found in {filename}")
                continue

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = splitter.split_documents(documents)
            if not docs:
                errors.append(f"❌ Failed to split documents in {filename}")
                continue

            persist_dir = os.path.join("chroma_db", filename)
            os.makedirs(persist_dir, exist_ok=True)

            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
            vectorstores.append(vectorstore)

            uploaded.append(filename)
        except Exception as e:
            errors.append(f"❌ Error processing {filename}: {str(e)}")

    message = f"✅ Uploaded: {', '.join(uploaded)}" if uploaded else "❌ No files uploaded."
    return jsonify({"message": message, "errors": errors})

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "Missing question"}), 400
    if not vectorstores:
        return jsonify({"error": "No PDFs uploaded"}), 400

    try:
        all_docs = []
        for store in vectorstores:
            docs = store.as_retriever().invoke(question)
            all_docs.extend(docs)

        if not all_docs:
            return jsonify({"answer": "Sorry, I don't have information on that.", "time": "0.00s"})

        context = "\n\n".join(doc.page_content[:1000] for doc in all_docs[:5])  # More context for richer response

        prompt = f"""
You are a highly knowledgeable and helpful maritime assistant. Answer the user's question in detailed paragraphs using ONLY the following context.
Avoid generic responses or saying \"According to the document.\"
If the answer is not found, say \"Sorry, I don't have information on that.\"

Use the following structure:
1. Begin with a clear and complete answer.
2. Provide reasoning or explanation in a technical yet understandable manner.
3. If applicable, list procedures or parts in bullet points.

Context:
{context}

Question:
{question}

Answer:
"""

        start = time.time()
        llm = OllamaLLM(model="llama3")
        response = llm.invoke(prompt)
        duration = time.time() - start

        if not response:
            return jsonify({"answer": "Sorry, I couldn't generate a response.", "time": f"{duration:.2f}s"})

        return jsonify({"answer": response.strip(), "time": f"{duration:.2f}s"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
