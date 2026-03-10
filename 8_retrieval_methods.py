import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()

    print("======================================================================")
    print("=== RAG Advanced Retrieval Methods (FAISS) ===")
    print("======================================================================\n")

    # 1. SETUP & LOAD DATABASE
    persistent_directory = "db/faiss_db"
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print(f"Loading FAISS database from {persistent_directory}...")
    # allow_dangerous_deserialization is required for loading local FAISS files
    db = FAISS.load_local(
        persistent_directory, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )

    query = "How much did Microsoft pay to acquire GitHub?"
    print(f"\nUser Query: {query}")
    print("-" * 60)

    # ──────────────────────────────────────────────────────────────────
    # METHOD 1: Basic Similarity Search
    # Best for: Standard questions where you just want the closest matches.
    # ──────────────────────────────────────────────────────────────────
    # print("\n[METHOD 1: SIMILARITY SEARCH]")
    # print("Goal: Find the top 3 most mathematically similar chunks.")
    
    # retriever_sim = db.as_retriever(search_kwargs={"k": 3})
    # docs_sim = retriever_sim.invoke(query)
    
    # for i, doc in enumerate(docs_sim, 1):
    #     print(f"  Match {i}: {doc.page_content[:250]}...")


    # # ──────────────────────────────────────────────────────────────────
    # # METHOD 2: Similarity with Score Threshold
    # # Best for: High-stakes info. If the "best" match is still garbage, return nothing.
    # # ──────────────────────────────────────────────────────────────────
    # print("\n[METHOD 2: SCORE THRESHOLD]")
    # print("Goal: Only return chunks if they are relevant enough (Threshold: 0.5).")
    
    # # Note: For FAISS, a higher threshold means stricter matching.
    # retriever_thresh = db.as_retriever(
    #     search_type="similarity_score_threshold",
    #     search_kwargs={"k": 3, "score_threshold": 0.3}
    # )
    
    # docs_thresh = retriever_thresh.invoke(query)
    # if not docs_thresh:
    #     print("  ⚠️ No documents met the 0.3 similarity threshold.")
    # for i, doc in enumerate(docs_thresh, 1):
    #     print(f"  Match {i}: {doc.page_content[:250]}...")


    # # ──────────────────────────────────────────────────────────────────
    # # METHOD 3: Maximum Marginal Relevance (MMR)
    # # Best for: Summarization. It picks the most relevant doc, then picks 
    # # the next one because it is relevant AND different from the first.
    # # ──────────────────────────────────────────────────────────────────
    print("\n[METHOD 3: MAXIMUM MARGINAL RELEVANCE (MMR)]")
    print("Goal: Find relevant documents that don't repeat the same info.")
    
    retriever_mmr = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,           # Final docs to return
            "fetch_k": 10,    # Initial pool to look at before diversifying
            "lambda_mult": 0.5 # 0 = Max Diversity, 1 = Max Relevance
        }
    )
    
    docs_mmr = retriever_mmr.invoke(query)
    for i, doc in enumerate(docs_mmr, 1):
        print(f"  Match {i}: {doc.page_content[:250]}...")

    print("\n======================================================================")
    print("✅ Retrieval Comparison Complete!")

if __name__ == "__main__":
    main()