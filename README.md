# Resource-Aware LLM Router (OpenAI Responses API + Web Search)

A small CLI app that routes user questions to different OpenAI models based on the **type of prompt**. It uses the **OpenAI Responses API** and (when needed) the built-in **Web Search tool** to answer questions that require current information.

---

## What It Does

1. Prompts the user for a question in the terminal.
2. Classifies the question into one of three categories:
   - **simple** — direct factual questions
   - **reasoning** — multi-step logic/math/inference
   - **internet_search** — questions requiring up-to-date web information
3. Routes the question to an appropriate model:
   - `gpt-4.1-mini` for **simple**
   - `o4-mini` for **reasoning**
   - `gpt-4.1-mini` + `web_search` tool for **internet_search**
4. Prints:
   - classification
   - model used
   - final response

---

## Why This Is Useful

- Avoids using expensive/slow models when they aren’t needed
- Automatically triggers web search for questions involving “today”, “latest”, or specific current-year requests
- Provides a clear structure for building “resource-aware” assistants

---

## Requirements

- Python 3.11+ (3.12 recommended)
- OpenAI API key

### Python dependencies

- `openai`
- `python-dotenv`

---

## Setup

### 1) Create and activate a virtual environment (Windows / PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

python -m pip install -U openai python-dotenv

python Resource_Aware_LLM.py
