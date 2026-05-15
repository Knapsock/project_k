from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import os
import time

app = Flask(__name__)

combined_docs = []
combined_vectorstore = None
qa_chain = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    global combined_docs, combined_vectorstore, qa_chain

    files = request.files.getlist("pdfs")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    os.makedirs("uploads", exist_ok=True)

    for file in files:
        if file.filename.endswith(".pdf"):
            path = os.path.join("uploads", file.filename)
            file.save(path)

            loader = PyPDFLoader(path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            combined_docs.extend(chunks)

    # Recreate vectorstore with all combined chunks
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    combined_vectorstore = Chroma.from_documents(combined_docs, embedding=embeddings, persist_directory="chroma_db_combined")

    llm = OllamaLLM(model="llama3")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=combined_vectorstore.as_retriever(),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=3)
    )

    return jsonify({"message": f"{len(files)} PDF(s) uploaded and indexed."})

@app.route("/ask", methods=["POST"])
def ask_question():
    global qa_chain

    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "Question required"}), 400
    if not qa_chain:
        return jsonify({"error": "Upload PDFs first"}), 400

    try:
        start = time.time()
        result = qa_chain.invoke({"question": question})
        duration = time.time() - start
        return jsonify({"answer": result["answer"], "time": f"{duration:.2f}s"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
