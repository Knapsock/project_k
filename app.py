from flask import Flask, request, render_template
import fitz  # PyMuPDF
import os
import pickle
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.quiz_gen import generate_quiz_from_chunks

app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_STORE = "vector_store/pdf_vectors.pkl"

# Extract text from all pages (without page numbers)
def extract_pdf_text(pdf_file):
    text = []
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text.append(page.get_text())
    return text

# Chunk text (no metadata)
def chunk_text(pages, chunk_size=300):
    chunks = []
    for page_text in pages:
        for i in range(0, len(page_text), chunk_size):
            chunk = page_text[i:i + chunk_size]
            chunks.append(chunk)
    return chunks

# Embed chunks and save to vector store
def embed_and_store_chunks(chunks):
    embeddings = model.encode(chunks)
    os.makedirs("vector_store", exist_ok=True)
    with open(VECTOR_STORE, "wb") as f:
        pickle.dump((chunks, embeddings), f)

# Load chunks and embeddings safely (support old 3-value format too)
def load_vector_store():
    with open(VECTOR_STORE, "rb") as f:
        data = pickle.load(f)
        if isinstance(data, tuple) and len(data) == 3:
            chunks, embeddings, _ = data  # discard metadata
        else:
            chunks, embeddings = data
    return chunks, embeddings

# Get best answer from top-k similar chunks
def get_best_answer(question, top_k=3):
    chunks, embeddings = load_vector_store()
    question_embedding = model.encode([question])
    similarities = cosine_similarity(question_embedding, embeddings)[0]

    top_indices = similarities.argsort()[-top_k:][::-1]
    selected_chunks = [chunks[i] for i in top_indices]

    context = "\n\n".join(selected_chunks)

    prompt = f"""
You are a helpful assistant. Read the content and answer the user's question with an accurate explanation. 
Then, explain the answer in simple, human-understandable language that even a non-expert can grasp.

Avoid phrases like "According to the document" or "Based on the PDF." Be direct and clear.

If the answer is unclear, respond with:
"Sorry, i DONT HAVE ANY INFORMATION OF THAT KIND."

Steps:
1. Analyze the provided content.
2. Answer the question based on facts.
3. Then rephrase the answer in a simpler, easy-to-understand version.

Content:
{context}

Question:
{question}

Answer:
"""


    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    if response.status_code == 200:
        return response.json()["response"].strip()
    else:
        return "⚠️ Error: Could not get a response from the LLM."

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        question = request.form.get("question", "")
        pdf = request.files.get("pdf")

        if pdf:
            pages = extract_pdf_text(pdf)
            chunks = chunk_text(pages)
            embed_and_store_chunks(chunks)
            answer = "✅ PDF uploaded and processed. You can now ask questions or generate a quiz!"
        elif question:
            if not os.path.exists(VECTOR_STORE):
                answer = "⚠️ Please upload a PDF first."
            else:
                answer = get_best_answer(question)
        else:
            answer = "Please enter a question or upload a PDF."

    return render_template("index.html", answer=answer)

@app.route("/quiz")
def quiz():
    chunks, _ = load_vector_store()
    quiz_data = generate_quiz_from_chunks(chunks)
    return render_template("quiz.html", quiz=quiz_data)

if __name__ == "__main__":
    app.run(debug=True)

