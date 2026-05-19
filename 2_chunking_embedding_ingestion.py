import os
import shutil
import time
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def main():
    file_path = "source_text.txt"
    db_directory = "./chroma_db"
    collection_name = "local_rag_collection"

    print("🚀 Script started...")

    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found.")
        return

    if os.path.exists(db_directory):
        print("🧹 Removing old database instance...")
        shutil.rmtree(db_directory)

    print("📄 Loading raw text data...")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    print("✂️ Splitting text into safe, optimized chunks (2000 chars)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=250,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    total_chunks = len(chunks)
    print(f"🧩 Created {total_chunks} text chunks.")

    print("🧠 Connecting to Local Nomic Engine...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("📥 Ingesting chunks into ChromaDB...")
    # ⚡ INCREASED TO 100: For maximum parallel speed
    batch_size = 100
    vector_store = None

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        current_max = min(i + batch_size, total_chunks)
        
        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch, 
                embedding=embeddings, 
                persist_directory=db_directory,
                collection_name=collection_name
            )
        else:
            vector_store.add_documents(documents=batch)
            
        percent_done = int((current_max / total_chunks) * 100)
        print(f"🚀 Speed Run Progress: [{percent_done}%] -> Ingested {current_max}/{total_chunks} chunks.")
        
        # ⚡ REMOVED THE SLEEP TIMER COMPLETELY!
        
    print(f"✅ Success! Saved all {total_chunks} embeddings cleanly to {db_directory}!")

if __name__ == "__main__":
    main()
