from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# Load PDF
loader = PyPDFLoader("dvsdv.pdf")  # Ensure the filename is correct and in the same directory
pages = loader.load()

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

# Create vector DB from chunks using embeddings
embedding = OllamaEmbeddings(model="nomic-embed-text")  # Ensure this embedding model is pulled
vectordb = Chroma.from_documents(chunks, embedding=embedding, persist_directory="./vector_store")
retriever = vectordb.as_retriever()

# Load LLM (Ollama model must be running and available)
llm = OllamaLLM(model="llama3")  # Or "my-kbot" if using your own custom model

# Build RetrievalQA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Interactive terminal Q&A
print("📘 Ask questions about your PDF. Type 'exit' to quit.")
while True:
    query = input("\nYour question: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = qa.run(query)
    print(f"\n🤖 Answer: {response}")
