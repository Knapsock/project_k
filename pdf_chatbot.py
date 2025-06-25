import os
import random
import sqlite3
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog, Toplevel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import time
import json
import re
import matplotlib.pyplot as plt
from PIL import ImageTk, Image

class MaritimeChatbot:
    def __init__(self, root):
        self.is_docker = os.environ.get('IS_DOCKER', False)
        if self.is_docker:
            os.makedirs("/data", exist_ok=True)
            os.chdir("/data")
        else:
            os.environ.pop("DISPLAY", None)

        self.root = root
        self.root.title("Maritime Expert Assistant")
        self.root.geometry("1000x750")
        self.theme = "light"
        self.configure_theme()

        self.qa_chain = None
        self.current_pdf = None
        self.mcq_bank = []
        self.mcq_mode = False
        self.current_question_index = 0
        self.user_score = 0
        self.user_logs = []
        self.retriever = None

        self.setup_ui()
        self.setup_database()
        self.load_sample_pdf()

        if not self.check_ollama_connection():
            messagebox.showerror("Ollama Error", "Failed to connect to Ollama service")

    def check_ollama_connection(self):
        try:
            import ollama
            ollama.list()
            return True
        except Exception as e:
            self.display_message(f"Ollama connection failed: {str(e)}", "error")
            return False

    def configure_theme(self):
        if self.theme == "dark":
            self.bg_color = "#1e1e1e"
            self.fg_color = "#ffffff"
            self.chat_bg = "#2e2e2e"
            self.button_bg = "#0078D7"
        else:
            self.bg_color = "#f0f2f5"
            self.fg_color = "#000000"
            self.chat_bg = "#ffffff"
            self.button_bg = "#0078D7"
        self.root.configure(bg=self.bg_color)

    def setup_ui(self):
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background=self.button_bg, 
                       foreground="white", font=("Arial", 11, "bold"))
        style.configure("TLabel", font=("Arial", 12), background=self.bg_color, foreground=self.fg_color)

        title = ttk.Label(self.root, text="🚢 Maritime Expert Assistant", font=("Helvetica", 18, "bold"), anchor="center")
        title.pack(pady=10)

        self.chat_display = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, font=('Arial', 12), bg=self.chat_bg, fg=self.fg_color, height=25
        )
        self.chat_display.tag_config("user", foreground="#1a73e8")
        self.chat_display.tag_config("bot", foreground="#0d652d")
        self.chat_display.tag_config("error", foreground="#d93025")
        self.chat_display.tag_config("debug", foreground="#666666")
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
            ("🔄 Retry MCQs", self.retry_mcq_generation),
            ("🌃 Toggle Theme", self.toggle_theme),
            ("🏆 Leaderboard", self.show_leaderboard),
            ("📊 Quiz Stats", self.show_quiz_stats)
        ]

        for text, cmd in buttons:
            ttk.Button(btn_frame, text=text, command=cmd).pack(side=tk.LEFT, padx=5)

    def toggle_theme(self):
        self.theme = "dark" if self.theme == "light" else "light"
        self.configure_theme()
        self.chat_display.config(bg=self.chat_bg, fg=self.fg_color)
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Frame):
                widget.config(bg=self.bg_color)

    def display_message(self, msg, sender="bot"):
        tag = sender
        self.chat_display.insert(tk.END, f"{msg}\n", tag)
        self.chat_display.see(tk.END)
        self.root.update_idletasks()

    def process_input(self, event=None):
        question = self.user_input.get()
        self.user_input.delete(0, tk.END)
        if not question:
            return
        self.display_message(f"You: {question}", "user")
        try:
            if self.qa_chain:
                start = time.time()
                result = self.qa_chain.invoke({"question": question})
                duration = time.time() - start
                answer = result["answer"]
                self.display_message(f"Assistant: {answer} (⏱️ {duration:.2f}s)", "bot")

                if any(keyword in question.lower() for keyword in ["diagram", "draw", "sketch", "flowchart"]):
                    self.generate_diagram()

            else:
                self.display_message("Please load a PDF first.", "error")
        except Exception as e:
            self.display_message(f"Error: {str(e)}", "error")

    def generate_diagram(self):
        try:
            fig, ax = plt.subplots()
            ax.set_title("Simple Ship Diagram")
            ship = plt.Rectangle((0.2, 0.4), 0.6, 0.2, color='steelblue')
            ax.add_patch(ship)
            ax.annotate("Ship Hull", (0.5, 0.5), color='white', ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

            img_path = "diagram.png"
            fig.savefig(img_path, bbox_inches='tight')
            plt.close(fig)

            top = Toplevel(self.root)
            top.title("Diagram Viewer")
            img = ImageTk.PhotoImage(Image.open(img_path))
            lbl = ttk.Label(top, image=img)
            lbl.image = img
            lbl.pack()

        except Exception as e:
            self.display_message(f"Error generating diagram: {e}", "error")

    def load_pdf(self):
        initial_dir = "/data" if self.is_docker else None
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")], initialdir=initial_dir)
        if not file_path:
            return

        try:
            self.display_message("Processing PDF... Please wait.", "bot")
            self.root.update()

            loader = PyPDFLoader(file_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = splitter.split_documents(documents)

            persist_dir = os.path.join(".", "chroma_db")
            os.makedirs(persist_dir, exist_ok=True)

            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)

            llm = OllamaLLM(model="llama3")
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=3)
            )
            self.retriever = vectorstore.as_retriever()
            self.current_pdf = os.path.basename(file_path)
            self.display_message(f"✅ PDF '{self.current_pdf}' loaded successfully!", "bot")
            self.display_message("You can now ask questions or generate MCQs.", "bot")

        except Exception as e:
            self.display_message(f"Error loading PDF: {str(e)}", "error")

    def setup_database(self):
        pass

    def load_sample_pdf(self):
        pass

    def start_mcq_quiz(self):
        pass

    def retry_mcq_generation(self):
        pass

    def show_leaderboard(self):
        pass

    def show_quiz_stats(self):
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = MaritimeChatbot(root)
    root.mainloop()