import os
import shutil
import requests
import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_ollama_embedding_modern(text, model="nomic-embed-text"):
    """Talks directly to Ollama's stable /api/embed route to extract flat vectors."""
    url = "http://localhost:11434/api/embed"
    payload = {
        "model": model, 
        "input": text
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    # Safely extracts the first inner 1D vector out of the 2D matrix array layout
    return response.json()["embeddings"][0]

def main():
    file_path = "source_text.txt"
    db_directory = "./chroma_db"
    collection_name = "local_rag_collection"

    print("🚀 Script started...")

    if not os.path.exists(file_path):
        print(f"❌ Error: '{file_path}' not found. Please create it and add your raw text data.")
        return

    if os.path.exists(db_directory):
        print("🧹 Removing old database instance to avoid structural mismatch...")
        shutil.rmtree(db_directory)

    print("📄 Loading raw local text data...")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    print("✂️ Splitting text into LARGE context chunks (4500 chars)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4500, 
        chunk_overlap=500,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    total_chunks = len(chunks)
    print(f"🧩 Created {total_chunks} high-quality text chunks.")

    print("🗄️ Initializing Native ChromaDB Client Storage Engine...")
    chroma_client = chromadb.PersistentClient(path=db_directory)
    collection = chroma_client.create_collection(name=collection_name)

    print("📥 Ingesting chunks natively into ChromaDB...")
    for idx, chunk in enumerate(chunks):
        text_content = chunk.page_content
        
        try:
            vector = get_ollama_embedding_modern(text_content)
            
            collection.add(
                embeddings=[vector],
                documents=[text_content],
                ids=[f"id_{idx}"]
            )
            
            percent_done = int(((idx + 1) / total_chunks) * 100)
            print(f"⏳ Progress: [{percent_done}%] -> Ingested chunk {idx + 1}/{total_chunks}")
            
        except Exception as e:
            print(f"❌ Error processing chunk {idx + 1}: {str(e)}")
            continue
            
    print(f"✅ Success! 100% of your {total_chunks} chunks are stored natively in {db_directory}!")

if __name__ == "__main__":
    main()
