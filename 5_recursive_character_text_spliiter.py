import os
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

def main():
    print("======================================================================")
    print("=== RAG Chunking Strategies Demonstration ===")
    print("======================================================================\n")

    # The sample text with a mix of short lines and one massive paragraph
    tesla_text = """Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance

The Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production Challenges

Supply chain issues caused a 12% increase in production costs.

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever making it impossible to split properly."""

    print("--- 1. CHARACTER TEXT SPLITTER (The Brute Force Way) ---")
    print("Notice how it struggles with the massive paragraph at the end because it only looks for specific separators.\n")
    
    splitter_basic = CharacterTextSplitter(
        separator=" ",  # Default separator
        chunk_size=100,
        chunk_overlap=0
    )

    chunks_basic = splitter_basic.split_text(tesla_text)
    for i, chunk in enumerate(chunks_basic, 1):
        print(f"Chunk {i}: ({len(chunk)} chars)")
        print(f'"{chunk}"\n')


    print("======================================================================")
    print("--- 2. RECURSIVE CHARACTER TEXT SPLITTER (The Smart Way) ---")
    print("Notice how it gracefully breaks down the text by checking paragraphs, then sentences, then words.\n")

    recursive_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],  # Falls back through these separators in order
        chunk_size=100,
        chunk_overlap=0
    )

    chunks_recursive = recursive_splitter.split_text(tesla_text)
    for i, chunk in enumerate(chunks_recursive, 1):
        print(f"Chunk {i}: ({len(chunk)} chars)")
        print(f'"{chunk}"\n')

    print("======================================================================")
    print("✅ Demonstration Complete!")
    print("Note: You are already using RecursiveCharacterTextSplitter in your document ingestion pipeline!")

if __name__ == "__main__":
    main()