import os
import shutil
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def main():
    print("=== RAG Document Ingestion Pipeline ===")
    docs_path = "docs"
    persistent_directory = "db/faiss_db" 
    
    if os.path.exists(persistent_directory):
        print("Cleaning up old database...")
        shutil.rmtree(persistent_directory)

    # 1. Load
    print(f"\nLoading documents from {docs_path}...")
    loader = DirectoryLoader(path=docs_path, glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()

    # 2. Split
    print("Splitting documents safely...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Successfully created {len(chunks)} safe chunks.")

    # 3. Create FAISS Vector Store
    print("\nSending to OpenAI and saving FAISS database (No Windows crashing!)...")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # FAISS processes this natively without the SQLite bug
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(persistent_directory)
    
    print(f"\n✅ Ingestion complete! Database saved to {persistent_directory}.")

if __name__ == "__main__":
    main()