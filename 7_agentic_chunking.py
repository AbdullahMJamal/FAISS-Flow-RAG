import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()

    print("======================================================================")
    print("=== RAG Agentic (LLM-Based) Chunking Demonstration ===")
    print("======================================================================\n")

    # The Tesla sample text
    tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""

    # 1. SETUP THE "AGENTIC" BRAIN
    # We use temperature=0 because we want the AI to be precise, not creative.
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    print("🤖 Agent is analyzing the text for logical boundaries...")

    # 2. CREATE THE META-PROMPT
    # This instruction set turns the LLM into a specialized chunking tool.
    prompt_text = f"""
    You are a text chunking expert. Split this text into logical chunks.

    Rules:
    - Each chunk should be around 200 characters or less.
    - Split at natural topic boundaries (don't cut a thought in half).
    - Keep related information together.
    - Put "<<<SPLIT>>>" between chunks.

    Text:
    {tesla_text}

    Return the text with <<<SPLIT>>> markers where you want to split:
    """

    # 3. GET THE AI TO ADD MARKERS
    response = llm.invoke([HumanMessage(content=prompt_text)])
    marked_text = response.content

    # 4. PARSE THE MARKERS INTO A LIST
    # This splits the single string into a list based on the AI's markers
    raw_chunks = marked_text.split("<<<SPLIT>>>")
    
    # Clean up whitespace for a professional finish
    clean_chunks = [c.strip() for c in raw_chunks if c.strip()]

    # 5. DISPLAY RESULTS
    print("\n🎯 AGENTIC CHUNKING RESULTS:")
    print("=" * 50)

    for i, chunk in enumerate(clean_chunks, 1):
        print(f"Chunk {i}: ({len(chunk)} chars)")
        print(f'"{chunk}"')
        print("-" * 30)

    print("\n======================================================================")
    print("✅ Agentic Chunking Complete!")
    print("Note: This is the most accurate method, but it costs the most API credits.")

if __name__ == "__main__":
    main()