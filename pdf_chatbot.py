import os
import random
import sqlite3
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog, Toplevel
from PIL import Image, ImageTk
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import matplotlib.pyplot as plt
from io import BytesIO


class MaritimeChatbot:
    def __init__(self, root):
        self.root = root
        self.root.title("Maritime Expert Assistant")
        self.root.geometry("1000x750")
        self.theme = "light"
        self.configure_theme()

        self.qa_chain = None
        self.current_pdf = None
        self.mcq_bank = self._load_mcq_database()
        self.mcq_mode = False
        self.current_question_index = 0
        self.user_score = 0

        self.setup_ui()
        self.setup_database()
        self.load_sample_pdf()

    def configure_theme(self):
        if self.theme == "dark":
            self.bg_color = "#1e1e1e"
            self.fg_color = "#ffffff"
            self.chat_bg = "#2e2e2e"
        else:
            self.bg_color = "#f0f2f5"
            self.fg_color = "#000000"
            self.chat_bg = "#ffffff"
        self.root.configure(bg=self.bg_color)

    def toggle_theme(self):
        self.theme = "dark" if self.theme == "light" else "light"
        self.configure_theme()
        self.root.destroy()
        root = tk.Tk()
        MaritimeChatbot(root)
        root.mainloop()

    def setup_ui(self):
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#0078D7", foreground="white", font=("Arial", 11, "bold"))
        style.configure("TLabel", font=("Arial", 12), background=self.bg_color, foreground=self.fg_color)

        title = ttk.Label(self.root, text="😳 Maritime Expert Assistant", font=("Helvetica", 18, "bold"), anchor="center")
        title.pack(pady=10)

        self.chat_display = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, font=('Arial', 12), bg=self.chat_bg, fg=self.fg_color, height=25
        )
        self.chat_display.pack(padx=15, pady=10, fill=tk.BOTH, expand=True)

        input_frame = tk.Frame(self.root, bg=self.bg_color)
        input_frame.pack(padx=15, pady=5, fill=tk.X)

        self.user_input = tk.Entry(input_frame, font=('Arial', 13))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", self.process_input)

        ttk.Button(input_frame, text="💬 Ask", command=lambda: self.process_input(None)).pack(side=tk.RIGHT)

        btn_frame = tk.Frame(self.root, bg=self.bg_color)
        btn_frame.pack(pady=10)

        buttons = [
            ("📁 Load PDF", self.load_pdf),
            ("🧠 MCQ Quiz", self.start_mcq_quiz),
            ("🌓 Toggle Theme", self.toggle_theme),
            ("🏆 Leaderboard", self.show_leaderboard)
        ]

        for text, cmd in buttons:
            ttk.Button(btn_frame, text=text, command=cmd).pack(side=tk.LEFT, padx=6)

    def load_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if not file_path:
            return

        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = splitter.split_documents(documents)

            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=OllamaLLM(model="mistral"),
                retriever=vectorstore.as_retriever(),
                memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            )
            self.display_message("PDF loaded and processed successfully!", "bot")
        except Exception as e:
            self.display_message(f"Error loading PDF: {e}", "error")

    def load_sample_pdf(self):
        pass

    def generate_diagram(self, question):
        if "engine" in question.lower():
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "[Engine Diagram]", ha='center', va='center', fontsize=14)
            ax.axis('off')
        elif "scavenging" in question.lower():
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "[Scavenging Process]", ha='center', va='center', fontsize=14)
            ax.axis('off')
        else:
            return

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img = ImageTk.PhotoImage(img)

        img_label = tk.Label(self.chat_display, image=img, bg=self.chat_bg)
        img_label.image = img
        self.chat_display.window_create(tk.END, window=img_label)
        self.chat_display.insert(tk.END, "\n")

    def process_input(self, event):
        question = self.user_input.get()
        self.user_input.delete(0, tk.END)

        if not question:
            return
        self.display_message(f"You: {question}", "user")

        if self.mcq_mode:
            self.handle_mcq_response(question)
            return

        try:
            if self.qa_chain:
                result = self.qa_chain({"question": question})
                answer = result["answer"]
                self.display_message(f"Assistant: {answer}", "bot")
                self.generate_diagram(question)
            else:
                self.display_message("Please load a PDF first", "error")
        except Exception as e:
            self.display_message(f"Error: {str(e)}", "error")

    def _load_mcq_database(self):
        return [
            {
                "question": "What is the function of a ship's ballast system?",
                "options": ["To steer the ship", "To control buoyancy", "To generate power", "To anchor the ship"],
                "answer": "To control buoyancy"
            },
            {
                "question": "Which part of the engine is responsible for air intake?",
                "options": ["Piston", "Cylinder", "Turbocharger", "Crankshaft"],
                "answer": "Turbocharger"
            },
            {
                "question": "What is scavenging in marine engines?",
                "options": [
                    "Collecting food from the sea",
                    "Removing exhaust gases and bringing in fresh air",
                    "Cleaning the ship",
                    "Repairing engine parts"
                ],
                "answer": "Removing exhaust gases and bringing in fresh air"
            }
        ]

    def start_mcq_quiz(self):
        self.mcq_mode = True
        self.current_question_index = 0
        self.user_score = 0
        self.display_message("Starting MCQ Quiz (3 questions):", "bot")
        self.ask_next_question()

    def ask_next_question(self):
        if self.current_question_index < 3:
            q = self.mcq_bank[self.current_question_index % len(self.mcq_bank)]
            options = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(q["options"])])
            self.display_message(f"Q{self.current_question_index+1}: {q['question']}\n{options}", "bot")
        else:
            self.display_message(f"Quiz completed! Your score: {self.user_score}/3", "bot")
            name = simpledialog.askstring("Username", "Enter your name for the leaderboard:") or "User"
            self.store_score(name, self.user_score)
            self.mcq_mode = False

    def handle_mcq_response(self, response):
        q = self.mcq_bank[self.current_question_index % len(self.mcq_bank)]
        try:
            selected = int(response.strip()) - 1
            if 0 <= selected < len(q["options"]):
                if q["options"][selected].lower() == q["answer"].lower():
                    self.display_message("Correct!", "bot")
                    self.user_score += 1
                else:
                    self.display_message(f"Wrong! Correct answer was: {q['answer']}", "bot")
                self.current_question_index += 1
                self.ask_next_question()
            else:
                self.display_message("Invalid option. Please select a number between 1 and 4.", "bot")
        except ValueError:
            self.display_message("Please enter a number (1-4).", "bot")

    def setup_database(self):
        self.conn = sqlite3.connect("leaderboard.db")
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS leaderboard (name TEXT, score INTEGER)''')
        self.conn.commit()

    def store_score(self, name, score):
        self.cursor.execute("INSERT INTO leaderboard (name, score) VALUES (?, ?)", (name, score))
        self.conn.commit()

    def show_leaderboard(self):
        leaderboard_win = tk.Toplevel(self.root)
        leaderboard_win.title("Leaderboard")
        leaderboard_win.geometry("300x300")

        self.cursor.execute("SELECT name, score FROM leaderboard ORDER BY score DESC LIMIT 10")
        entries = self.cursor.fetchall()

        tk.Label(leaderboard_win, text="Top Scores", font=("Helvetica", 14, "bold")).pack(pady=10)
        for i, (name, score) in enumerate(entries):
            tk.Label(leaderboard_win, text=f"{i+1}. {name} - {score}").pack()

    def display_message(self, msg, sender):
        tag = "user" if sender == "user" else "bot"
        self.chat_display.insert(tk.END, f"{msg}\n", tag)
        self.chat_display.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    MaritimeChatbot(root)
    root.mainloop()
