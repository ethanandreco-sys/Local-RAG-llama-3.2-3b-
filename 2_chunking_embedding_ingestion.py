import os
import shutil
import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def main():
    file_path = "source_text.txt"
    db_directory = "./chroma_db"
    collection_name = "local_rag_collection"

    print("🚀 Local Ingestion Script started...")

    if not os.path.exists(file_path):
        print(f"❌ Error: '{file_path}' not found. Please create it and add your raw text data.")
        return

    if os.path.exists(db_directory):
        print("🧹 Removing old database instance to avoid structural mismatch...")
        shutil.rmtree(db_directory)

    print("📄 Loading raw local text data...")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    print("✂️ Splitting text into context chunks...")
    # Kept your chunk size, but 4500 is very large. Consider 1000-2000 if your LLM struggles.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4500, 
        chunk_overlap=500,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"🧩 Created {len(chunks)} text chunks.")

    print("🧠 Initializing Local Ollama Embedding Driver...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("🗄️ Creating Vector Store and ingesting chunks locally...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_directory,
        collection_name=collection_name
    )
            
    print(f"✅ Success! All chunks are safely stored locally in {db_directory}!")

if __name__ == "__main__":
    main()