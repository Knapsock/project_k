import sqlite3
import random
import threading
import time
from PIL import Image
from tkinter import PhotoImage
import pytesseract
import speech_recognition as sr
import pyttsx3
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, simpledialog

# --- Voice I/O Setup ---
recognizer = sr.Recognizer()
engine = pyttsx3.init()
engine.setProperty('rate', 160)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    with sr.Microphone() as source:
        try:
            audio = recognizer.listen(source, timeout=5)
            return recognizer.recognize_google(audio)
        except:
            return ""

# Ask user if voice input should be enabled
enable_voice = messagebox.askyesno("Voice Input", "Do you want to enable voice input?")

# --- Load PDF ---
loader = PyPDFLoader("dvsdv.pdf")
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

embedding = OllamaEmbeddings(model="nomic-embed-text")
vectordb = Chroma.from_documents(chunks, embedding=embedding, persist_directory="./vector_store")
retriever = vectordb.as_retriever()

llm = OllamaLLM(model="meo-qa-v2")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# --- Load Quiz Questions ---
def load_quiz_questions(file_path="quiz_questions.txt"):
    questions = {"easy": [], "medium": [], "hard": [], "mcq_easy": []}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("[mcq_easy]"):
                    q = line[10:].strip()
                    options = [lines[i+1].strip(), lines[i+2].strip(), lines[i+3].strip(), lines[i+4].strip()]
                    answer_line = lines[i+5].strip()
                    correct = answer_line.split(":")[1].strip()
                    questions["mcq_easy"].append({
                        "question": q,
                        "options": options,
                        "answer": correct
                    })
                    i += 6
                elif line.startswith("[easy]"):
                    questions["easy"].append(line[6:].strip())
                    i += 1
                elif line.startswith("[medium]"):
                    questions["medium"].append(line[8:].strip())
                    i += 1
                elif line.startswith("[hard]"):
                    questions["hard"].append(line[6:].strip())
                    i += 1
                else:
                    i += 1
    except Exception as e:
        print(f"Error loading quiz: {e}")
    return questions

quiz_questions = load_quiz_questions()

# --- SQLite ---
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

# --- Image OCR ---
def process_image(image_path):
    try:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image).strip()
    except Exception as e:
        return f"Error: {e}"

# --- GUI Setup ---
def choose_avatar():
    global avatar_path
    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if path:
        avatar_path = path
        try:
            img = Image.open(avatar_path).resize((60, 60))
            avatar_img = ImageTk.PhotoImage(img)
            avatar_label.config(image=avatar_img)
            avatar_label.image = avatar_img  # Keep reference to avoid garbage collection
        except Exception as e:
            messagebox.showerror("Error", f"Could not load avatar: {e}")

app = tk.Tk()
app.title("PDF Chatbot")
app.geometry("700x600")
avatar_label = tk.Label(app)
avatar_label.pack(pady=(5, 0))

btn_avatar = tk.Button(app, text="👤 Choose Avatar", command=choose_avatar)
btn_avatar.pack(pady=(0, 10))

user_input = tk.StringVar()

# Theme toggle logic
def toggle_theme():
    global dark_mode
    dark_mode = not dark_mode
    apply_theme()

def apply_theme():
    bg_color = "#2e2e2e" if dark_mode else "#ffffff"
    fg_color = "#ffffff" if dark_mode else "#000000"

    app.configure(bg=bg_color)
    chat_box.configure(bg=bg_color, fg=fg_color, insertbackground=fg_color)
    entry.configure(bg=bg_color, fg=fg_color, insertbackground=fg_color)
    for child in frame.winfo_children():
        try:
            child.configure(bg=bg_color, fg=fg_color)
        except:
            pass

user_name = simpledialog.askstring("User Name", "Enter your name")
avatar_path = None  # Global to store the avatar image path


chat_box = scrolledtext.ScrolledText(app, wrap=tk.WORD)
chat_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

entry = tk.Entry(app, font=("Arial", 14))
entry.pack(fill=tk.X, padx=10, pady=5)

frame = tk.Frame(app)
frame.pack(pady=10)

btn_mcq = tk.Button(frame, text="📝 MCQ Quiz", command=lambda: start_mcq_quiz())
btn_mcq.grid(row=1, column=0, padx=5)

btn_timer = tk.Label(frame, text="⏳")
btn_timer.grid(row=1, column=1, padx=5)

# --- MCQ Quiz Handler ---
def start_mcq_quiz():
    mode = simpledialog.askstring("Quiz Mode", "Select difficulty: easy / medium / hard")
    if mode not in quiz_questions:
        chat_box.insert(tk.END, "Invalid mode. Please choose: easy, medium, or hard.\n")
        return

    questions = quiz_questions[mode]
    if len(questions) < 15:
        chat_box.insert(tk.END, "Not enough questions in this mode.\n")
        return

    score = 0
    attempted = 0
    total = 15
    quiz_running = True

    def on_timeout():
        nonlocal quiz_running
        quiz_running = False
        timer_window.destroy()
        messagebox.showinfo("Timeout", "Time's up for this question!")

    for q in random.sample(questions, k=total):
        if not quiz_running:
            break

        # --- Parse Question ---
        try:
            question, *opts, correct = q.split("|")
        except:
            chat_box.insert(tk.END, f"Question format error: {q}\n")
            continue

        # --- Ask Question ---
        chat_box.insert(tk.END, f"\n🤖 (MCQ) {question.strip()}\n")
        opt_labels = ['A', 'B', 'C', 'D']
        options_dict = dict(zip(opt_labels, opts))

        for label, opt in options_dict.items():
            chat_box.insert(tk.END, f"{label}. {opt.strip()}\n")

        # --- Show Timer Window ---
        timer_window = tk.Toplevel(app)
        timer_window.title("⏰ Answer in 20 seconds")
        tk.Label(timer_window, text=question.strip(), font=("Arial", 12)).pack(padx=10, pady=5)

        var = tk.StringVar()
        for label in opt_labels:
            tk.Radiobutton(timer_window, text=f"{label}. {options_dict[label].strip()}",
                           variable=var, value=label, font=("Arial", 10)).pack(anchor='w')

        def submit_answer():
            timer_window.after_cancel(timer_id)
            timer_window.destroy()

        def end_quiz_early():
            nonlocal quiz_running
            quiz_running = False
            timer_window.destroy()

        # --- Buttons ---
        tk.Button(timer_window, text="Submit", command=submit_answer).pack(pady=5)
        tk.Button(timer_window, text="End Quiz", command=end_quiz_early).pack(pady=5)

        timer_id = timer_window.after(20000, on_timeout)  # 20 sec timer
        timer_window.mainloop()

        if not quiz_running:
            break

        answer = var.get().strip().upper()
        if not answer:
            chat_box.insert(tk.END, "⏳ You skipped this question or timed out.\n")
        else:
            attempted += 1
            if answer == correct.strip().upper():
                score += 1
                chat_box.insert(tk.END, "✅ Correct!\n")
            else:
                chat_box.insert(tk.END, f"❌ Wrong! Correct answer was {correct.strip().upper()}\n")

    chat_box.insert(tk.END, f"\n🧾 Quiz Ended\n")
    chat_box.insert(tk.END, f"Questions attempted: {attempted}/{total}\n")
    chat_box.insert(tk.END, f"Correct answers: {score}\n")
    chat_box.insert(tk.END, f"Final Score: {score}/{attempted if attempted else total}\n")

    if attempted > 0:
        save_score(user_name, score, attempted)


# --- Submit Handler ---
def submit(event=None):
    query = entry.get()
    entry.delete(0, tk.END)
    user_input.set(query)
    ask_question(query)

# --- Ask Question ---
def ask_question(query):
    if not query and enable_voice:
        query = listen()
        chat_box.insert(tk.END, f"🎤 You (voice): {query}\n")

    if not query:
        return

    chat_box.insert(tk.END, f"You: {query}\n")

    if query.lower() in ["exit", "quit"]:
        app.quit()
        return

    if query.lower() == "leaderboard":
    cursor.execute("""
    SELECT user, avatar, SUM(score) as total_score, COUNT(*) as attempts
    FROM scores GROUP BY user ORDER BY total_score DESC LIMIT 10
""")

    leaderboard = cursor.fetchall()
    chat_box.insert(tk.END, f"\n🏆 Leaderboard:\n")
    
    if avatar_path:
        chat_box.insert(tk.END, f"(Your Avatar Shown Below Leaderboard)\n")

    for row in leaderboard:
        chat_box.insert(tk.END, f"{row[0]} - Score: {row[1]} (Attempts: {row[2]})\n")

    if avatar_path:
        try:
            img = Image.open(avatar_path).resize((80, 80))
            avatar_img = ImageTk.PhotoImage(img)
            avatar_display = tk.Label(app, image=avatar_img)
            avatar_display.image = avatar_img
            avatar_display.pack(pady=5)
        except Exception as e:
            chat_box.insert(tk.END, f"Avatar load error: {e}\n")
    return


    try:
        response = qa.invoke({"question": query})["answer"]
        chat_box.insert(tk.END, f"🤖: {response}\n")
    except Exception as e:
        chat_box.insert(tk.END, f"Error: {e}\n")

btn_theme = tk.Button(frame, text="🌓 Theme", command=toggle_theme)
btn_theme.grid(row=1, column=2, padx=5)

btn_exit = tk.Button(frame, text="❌ Exit", command=app.quit)
btn_exit.grid(row=1, column=3, padx=5)

entry.bind("<Return>", submit)
entry.focus()

# Default theme
dark_mode = False
apply_theme()

app.mainloop()
