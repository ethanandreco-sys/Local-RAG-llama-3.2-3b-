import os
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

    print("📄 Loading raw local text data...")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    print("✂️ Splitting text into context chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"🧩 Created {len(chunks)} text chunks.")

    print("🧠 Initializing Local Ollama Embedding Driver...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("🧹 Force-clearing old collection data via Chroma API (Bypasses OS file locks)...")
    # This reaches inside the DB to explicitly delete the old collection name
    client = chromadb.PersistentClient(path=db_directory)
    try:
        client.delete_collection(name=collection_name)
        print("🗑️ Existing collection wiped clean.")
    except Exception:
        print("✨ No existing collection found. Starting fresh.")

    print("🗄️ Ingesting fresh text chunks into ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_directory,
        collection_name=collection_name
    )
            
    print(f"✅ Success! 100% refreshed chunks stored in {db_directory}!")

if __name__ == "__main__":
    main()