import os
from typing import List
from collections import defaultdict
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv

# 1. DEFINE STRUCTURED OUTPUT MODEL
class QueryVariations(BaseModel):
    queries: List[str] = Field(description="A list of 3 search query variations.")

# 2. DEFINE THE RRF FUNCTION
def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):
    """
    Combines multiple lists of retrieved documents into one ranked list.
    Formula: score = sum(1 / (k + rank))
    """
    if verbose:
        print("\n" + "="*60)
        print("APPLYING RECIPROCAL RANK FUSION (RRF)")
        print("="*60)
        print(f"Using k={k} | Calculating scores...\n")
    
    rrf_scores = defaultdict(float)  # {chunk_content: rrf_score}
    all_unique_chunks = {}           # {chunk_content: document_object}
    
    # Map for clean terminal output
    chunk_id_map = {}
    chunk_counter = 1
    
    for query_idx, chunks in enumerate(chunk_lists, 1):
        if verbose:
            print(f"Processing Results for Query Variation {query_idx}:")
        
        for position, chunk in enumerate(chunks, 1):
            content = chunk.page_content
            
            # Track unique chunks
            if content not in chunk_id_map:
                chunk_id_map[content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1
            
            chunk_id = chunk_id_map[content]
            all_unique_chunks[content] = chunk
            
            # The RRF Magic Formula: 1 / (k + rank)
            # Documents found in multiple searches will have their scores added together
            score_contribution = 1 / (k + position)
            rrf_scores[content] += score_contribution
            
            if verbose:
                print(f"  Pos {position}: {chunk_id} (+{score_contribution:.4f}) -> Total: {rrf_scores[content]:.4f}")
        
        if verbose: print()

    # Sort by highest score
    sorted_results = sorted(
        [(all_unique_chunks[content], score) for content, score in rrf_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    if verbose:
        print(f"✅ RRF Complete! Merged {len(chunk_lists)} searches into {len(sorted_results)} unique ranked documents.")
    
    return sorted_results

def main():
    load_dotenv()

    print("======================================================================")
    print("=== RAG Multi-Query + RRF Pipeline (FAISS) ===")
    print("======================================================================\n")

    # 3. SETUP & LOAD DATABASE
    persistent_directory = "db/faiss_db"
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print(f"Loading FAISS database from {persistent_directory}...")
    db = FAISS.load_local(
        persistent_directory, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )

    # 4. GENERATE VARIATIONS
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_structured = llm.with_structured_output(QueryVariations)

    original_query = "How does Tesla make money?"
    print(f"\nOriginal Query: {original_query}")
    
    prompt = f"Generate 3 variations of this query to improve retrieval: {original_query}"
    variations = llm_structured.invoke(prompt).queries

    print("Generated Variations:")
    for i, v in enumerate(variations, 1):
        print(f"  {i}. {v}")

    # 5. MULTI-RETRIEVAL
    retriever = db.as_retriever(search_kwargs={"k": 5})
    all_results = []

    for query in variations:
        all_results.append(retriever.invoke(query))

    # 6. APPLY FUSION
    fused_results = reciprocal_rank_fusion(all_results, k=60, verbose=True)

    # 7. DISPLAY TOP RESULTS
    print("\n" + "="*60)
    print("FINAL TOP RANKED CONTEXT")
    print("="*60)
    for rank, (doc, score) in enumerate(fused_results[:3], 1):
        print(f"RANK {rank} [Score: {score:.4f}]")
        print(f"{doc.page_content[:300]}...\n")

    print("======================================================================")
    print("Pipeline Success!")

if __name__ == "__main__":
    main()