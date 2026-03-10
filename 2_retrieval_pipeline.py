import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def main():
    print("======================================================================")
    print("=== RAG Retrieval Pipeline ===")
    print("======================================================================\n")
    
    persistent_directory = "db/faiss_db"
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("Loading FAISS database...")
    # allow_dangerous_deserialization is required by FAISS to load local files
    db = FAISS.load_local(persistent_directory, embedding_model, allow_dangerous_deserialization=True)
    
    query = "How much did Microsoft pay to acquire GitHub?"
    print(f"User Query: {query}\n")
    
    retriever = db.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(query)
    
    print("--- Context ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")

if __name__ == "__main__":
    main()