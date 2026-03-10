import os
from typing import List
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

# 1. DEFINE STRUCTURED OUTPUT MODEL
# This tells GPT-4o exactly how to format its response
class QueryVariations(BaseModel):
    queries: List[str] = Field(description="A list of 3 search query variations.")

def main():
    # Load environment variables
    load_dotenv()

    print("======================================================================")
    print("=== RAG Multi-Query Retrieval Pipeline (FAISS) ===")
    print("======================================================================\n")

    # 2. SETUP & LOAD DATABASE
    persistent_directory = "db/faiss_db"
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print(f"Loading FAISS database from {persistent_directory}...")
    db = FAISS.load_local(
        persistent_directory, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )

    # 3. INITIALIZE LLM WITH STRUCTURED OUTPUT
    # We use temperature=0 for consistency in query generation
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_structured_output = llm.with_structured_output(QueryVariations)

    # 4. GENERATE QUERY VARIATIONS
    original_query = "How does Tesla make money?"
    print(f"\nOriginal User Query: {original_query}")
    print("Generating search variations...")

    prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:
    Original query: {original_query}
    Return 3 alternative queries that rephrase or approach the same question from different angles."""

    response = llm_with_structured_output.invoke(prompt)
    query_variations = response.queries

    print("\n--- Generated Variations ---")
    for i, variation in enumerate(query_variations, 1):
        print(f"{i}. {variation}")

    print("\n" + "="*60)

    # 5. MULTI-QUERY RETRIEVAL
    # We search the database 3 separate times (one for each variation)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    all_retrieval_results = [] # To store results for later ranking (like RRF)

    for i, query in enumerate(query_variations, 1):
        print(f"\n Searching for Variation {i}: '{query}'")
        
        docs = retriever.invoke(query)
        all_retrieval_results.append(docs)
        
        print(f"Found {len(docs)} matching chunks:")
        for j, doc in enumerate(docs, 1):
            # Print first 100 characters of each match
            print(f"  Result {j}: {doc.page_content[:100]}...")
        
        print("-" * 50)

    print("\n======================================================================")
    print("✅ Multi-Query Retrieval Complete!")
    print(f"Total separate searches performed: {len(query_variations)}")

if __name__ == "__main__":
    main()