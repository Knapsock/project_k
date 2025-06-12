import sqlite3
import random
from PIL import Image
import pytesseract
import speech_recognition as sr
import pyttsx3
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- Voice I/O Setup ---
recognizer = sr.Recognizer()
engine = pyttsx3.init()
engine.setProperty('rate', 160)  # Speed of speech

def speak(text):
    print(f"🔊 Speaking: {text}")
    engine.say(text)
    engine.runAndWait()

def listen():
    with sr.Microphone() as source:
        print("🎤 Listening... (speak your question)")
        try:
            audio = recognizer.listen(source, timeout=5)
            query = recognizer.recognize_google(audio)
            print(f"🗣️ You said: {query}")
            return query
        except sr.UnknownValueError:
            print("❌ Could not understand audio.")
            return ""
        except sr.RequestError:
            print("⚠️ Could not request results.")
            return ""
        except sr.WaitTimeoutError:
            print("⏰ Listening timed out.")
            return ""

# --- Load PDF ---
loader = PyPDFLoader("dvsdv.pdf")
pages = loader.load()

# --- Split and Embed ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

embedding = OllamaEmbeddings(model="nomic-embed-text")
vectordb = Chroma.from_documents(chunks, embedding=embedding, persist_directory="./vector_store")
retriever = vectordb.as_retriever()

# --- LLM and Memory Setup ---
llm = OllamaLLM(model="meo-qa-v2")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# --- Load Quiz Questions from File ---
def load_quiz_questions(file_path="quiz_questions.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            questions = [line.strip() for line in file if line.strip()]
        return questions
    except FileNotFoundError:
        print("⚠️ quiz_questions.txt not found. Using default questions.")
        return [
            "What is B.O.D.?",
            "What are signs of water leakage in the turbocharger?",
            "How can debris be prevented from entering the engine system?"
        ]

quiz_questions = load_quiz_questions()

# --- SQLite Setup ---
conn = sqlite3.connect("quiz_scores.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT,
    score INTEGER,
    total INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

def save_score(user, score, total):
    cursor.execute("INSERT INTO scores (user, score, total) VALUES (?, ?, ?)", (user, score, total))
    conn.commit()

# --- Image Processing ---
def process_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"⚠️ Error processing image: {e}"

# --- Start Chat ---
print("💬 Ask any question related to the PDF.")
print("Type 'quiz' to test yourself, 'leaderboard' to view top scores, 'image:<path>' to analyze diagrams, 'voice' to use speech input, or 'exit' to quit.\n")

user_name = input("👤 Enter your name: ")

while True:
    query = input("\nYou: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    if query.lower() in ["hi", "hello", "hey"]:
        print("🤖: Hello! How can I help you today?")
        continue

    if query.lower() == "leaderboard":
        cursor.execute("""
        SELECT user, SUM(score) as total_score, COUNT(*) as attempts
        FROM scores
        GROUP BY user
        ORDER BY total_score DESC
        LIMIT 10
        """)
        rows = cursor.fetchall()
        print("\n🏆 Leaderboard (Top 10):")
        print(f"{'User':<15}{'Score':<10}{'Attempts'}")
        print("-" * 35)
        for row in rows:
            print(f"{row[0]:<15}{row[1]:<10}{row[2]}")
        continue

    if query.lower() == "quiz":
        print("\n🧠 Starting Quiz...")
        score = 0
        total = min(5, len(quiz_questions))
        questions_to_ask = random.sample(quiz_questions, k=total)

        for question in questions_to_ask:
            print(f"\n🤖 Quiz Question: {question}")
            user_answer = input("You: ")

            eval_prompt = f"""
You're an evaluator AI helping a student learn from a PDF document.

Compare the user's answer to the reference material and give feedback based on these:
- Is the meaning correct even if phrased differently?
- Is the key concept present?
- What part is wrong or missing, if any?
- Provide the ideal answer from the document.

### Question:
{question}

### User's Answer:
{user_answer}

### Your response format:
Correctness: Correct / Partially Correct / Incorrect  
Feedback: <Short explanation why>  
Ideal Answer: <Answer based on the PDF>
"""
            try:
                feedback = qa.invoke({"question": eval_prompt})["answer"]
                print("\n📝 Feedback:")
                print(feedback)

                if "correctness: correct" in feedback.lower():
                    score += 1
            except Exception as e:
                print(f"⚠️ Evaluation Error: {e}")

        print(f"\n🎯 You scored {score}/{total}.")
        save_score(user_name, score, total)

        if score == total:
            print("🏆 Perfect! You’ve mastered this topic.")
        elif score >= total // 2:
            print("👍 Nice! Keep refining your knowledge.")
        else:
            print("📚 Don't worry! Just shut up and study again.")
        continue

    if query.lower().startswith("image:"):
        image_path = query.split("image:")[1].strip()
        image_text = process_image(image_path)
        if image_text:
            try:
                response = qa.invoke({"question": f"Based on this image content: {image_text}"})["answer"]
                print(f"🤖: {response}")
            except Exception as e:
                print(f"⚠️ Error processing image content: {e}")
        continue

    if query.lower() == "voice":
        voice_query = listen()
        if voice_query:
            try:
                response = qa.invoke({"question": voice_query})["answer"]
                print(f"🤖: {response}")
                speak(response)
            except Exception as e:
                print(f"⚠️ Error with voice processing: {e}")
        continue

    try:
        response = qa.invoke({"question": query})["answer"]
        if response.strip():
            print(f"🤖: {response}")
        else:
            print("🤖: I couldn't find an answer in the PDF.")
    except Exception as e:
        print(f"⚠️ Error: {e}")
