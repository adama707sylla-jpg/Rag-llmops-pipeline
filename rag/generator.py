import os
import ollama
from dotenv import load_dotenv
 
load_dotenv()
 
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_URL  = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# Client Ollama pointant vers la bonne URL
client = ollama.Client(host=OLLAMA_URL)
 
SYSTEM_PROMPT = """You are a research assistant specialized in machine learning and AI.
You answer questions based ONLY on the provided research papers context.
Always cite the paper titles you used. Be concise and precise.
If the context doesn't contain enough information, say so clearly."""
 
 
def build_prompt(query: str, papers: list[dict]) -> str:
    context_parts = []
    for i, p in enumerate(papers, 1):
        context_parts.append(
            f"[Paper {i}] {p['title']}\n"
            f"Authors: {', '.join(p['authors'][:3])}\n"
            f"Abstract: {p['abstract'][:400]}...\n"
            f"URL: {p['url']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    return f"""Based on the following research papers:
 
{context}
 
---
 
Question: {query}
 
Answer based only on the papers above. Cite paper titles."""
 
 
def generate(query: str, papers: list[dict]) -> dict:
    if not papers:
        return {
            "answer"      : "No relevant papers found to answer this question.",
            "model"       : LLM_MODEL,
            "prompt_chars": 0,
        }
 
    prompt = build_prompt(query, papers)
 
    try:
        response = client.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        # response est un objet ChatResponse, pas un dict
        if hasattr(response, "message"):
            answer = response.message.content
        else:
            answer = response["message"]["content"]
 
    except ollama.ResponseError as e:
        raise RuntimeError(
            f"Ollama model error (model='{LLM_MODEL}'): {e.error}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Ollama unreachable — is it running on http://localhost:11434? ({e})"
        ) from e
 
    return {
        "answer"      : answer,
        "model"       : LLM_MODEL,
        "prompt_chars": len(prompt),
    }