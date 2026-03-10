import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

def main():
    print("======================================================================")
    print("=== RAG Retrieval & Generation Pipeline ===")
    print("======================================================================\n")
    
    # 1. SETUP & LOAD DATABASE
    persistent_directory = "db/faiss_db"
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("Loading FAISS database...")
    # Remember: We use FAISS instead of Chroma to prevent the Windows crash
    db = FAISS.load_local(
        persistent_directory, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    
    query = "How much did Microsoft pay to acquire GitHub?"
    print(f"\nUser Query: {query}\n")
    
    # 2. RETRIEVE (The "Search Engine" part)
    print("Searching for relevant documents...")
    retriever = db.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(query)
    
    print("--- Context Found ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}: {doc.page_content[:100]}...") # Just printing a preview so the terminal isn't cluttered

    # 3. GENERATE (The new "AI Brain" part)
    print("\nReading context and generating answer...")
    
    # This formats the documents into a clean list for the AI to read
    documents_text = "\n".join([f"- {doc.page_content}" for doc in relevant_docs])
    
    # This is the "Prompt". We give it strict instructions to ONLY use our documents.
    combined_input = f"""Based on the following documents, please answer this question: {query}

    Documents:
    {documents_text}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """

    # Create the ChatOpenAI model (The "Brain")
    # You can change "gpt-4o" to "gpt-3.5-turbo" if you want to save API credits!
    model = ChatOpenAI(model="gpt-4o")

    # Define the messages for the model
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]

    # Invoke the model with our prompt and documents
    result = model.invoke(messages)

    # Display the final human-readable answer!
    print("\n--- Generated Response ---")
    print(result.content)
    print("\n======================================================================")

if __name__ == "__main__":
    main()