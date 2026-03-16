import os
import requests
import json
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, Tuple

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")   

client = OpenAI()

#---Step 1: Classify the prompt for model selection---
def classify_prompt(prompt:str) -> dict:
    system_text = ("""
    You are a classifier that analyzes user prompts and returns one of the
    three categories ONLY: \n\n
    -Use 'simple' for the direct factual questions that need no reasoning or current events. \n
    -Use 'reasoning' for questions that require multi-step reasoning, problem-solving, or complex analysis. \n
    -Use 'internet search' if the prompt refers to current events, recent data, or other items not in your training data. \n
    Respond ONLY with a JSON like:\n
    '{"classification": "simple"}'
    """
    )

    resp = client.responses.create(
        model="gpt-5-mini",
        input = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": prompt}         
        ],
    )

    reply = (resp.output_text or "").strip()
    #Best effort JSON parse
    try:
        return json.loads(reply)
    except json.JSONDecodeError:
        # Try to extract the first JSON object from the response
        start = reply.find("{")
        end = reply.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(reply[start : end + 1])
        raise

#---Step 2: Web Search using OpenAI Responses API (Debugging Stage)---
def web_search(query: str, num_results: int=5) -> list[dict]:
    """Uses OpenAI's Responses API to perform as a web search
    tool to retrieve current information for the 'internet search' classification.
    Returns a list of {title, snippet, url}. Used in place of 
    Googles Custom Search API.
    NOTE: web_search is agentic; ask the model to format the results as JSON."""

    system_text = (
        """
You are a web research assistant. Use the web search tool to find relevant information
Return EXACTLY {num_results} if available
Respond ONLY with a valid JSON in this format:
'[{"title": "Title of the page", "snippet": "A brief summary of the page content", "url": "https://example.com"}, ...]'
   """ )
    resp = client.responses.create(
        model="gpt-5-mini",
        tools = [{"type": "web_search"}],
        input = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": query}         
        ],  
        temperature=0.0,
    )

    text = (resp.output_text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract the first JSON array from the response
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start : end + 1])
        else:
            return []
        
    return data.get("results", [])[:num_results]


# --- Step 3: Generate Response (Responses API + direct web_search tool) ---
def generate_response(prompt: str, classification: str) -> Tuple[str, str]:
    if classification == "simple":
        model = "gpt-4.1-mini"
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return (resp.output_text or ""), model

    if classification == "reasoning":
        model = "o4-mini"
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
        )
        return (resp.output_text or ""), model

    if classification == "internet_search":
        model = "gpt-4.1-mini"
        # Let the model run web search itself and answer with sources.
        resp = client.responses.create(
            model=model,
            tools=[{"type": "web_search"}],
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Use web search when needed. "
                        "When you use information from the web, cite sources by including the URL."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return (resp.output_text or ""), model

    # Fallback
    model = "gpt-4.1-mini"
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return (resp.output_text or ""), model

# --- Step 4: Combined Router ---
def handle_prompt(prompt: str) -> dict:
    classification_result = classify_prompt(prompt)
    classification = classification_result["classification"]

    answer, model = generate_response(prompt, classification)

    return {
        "classification": classification,
        "response": answer,
        "model": model,
    }


if __name__ == "__main__":
    user_prompt = input("Ask a question: ").strip()
    if not user_prompt:
        print("No question provided. Exiting.")
    else:
        result = handle_prompt(user_prompt)
        print("\n🔍 Classification:", result["classification"])
        print("🧠 Model Used:", result["model"])
        print("🧠 Response:\n", result["response"])