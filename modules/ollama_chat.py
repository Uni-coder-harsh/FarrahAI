"""
FarrahAI — Module 9: Ollama Integration
=========================================
Uses a locally running Ollama model to generate
well-formed answers from retrieved note chunks.

Ollama is the LAST step — it only formats and phrases the answer.
The knowledge comes from retrieved notes, NOT from the LLM's weights.

This keeps answers grounded in your uploaded notes.

Supported Ollama models (whatever you have installed):
  - mistral        (fast, good quality)
  - llama3         (high quality)
  - phi3           (very fast, lightweight)
  - gemma          (google, decent)
  - deepseek-r1    (if you have it)

Check what you have: run `ollama list` in terminal.
"""

import json
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "mistral"


def is_ollama_running(base_url: str = OLLAMA_BASE_URL) -> bool:
    """Check if Ollama server is running."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=3)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def list_available_models(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    """List all models installed in Ollama."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        logger.error(f"Could not list models: {e}")
        return []


def chat(prompt: str,
         model: str = DEFAULT_MODEL,
         system_prompt: Optional[str] = None,
         base_url: str = OLLAMA_BASE_URL,
         timeout: int = 120) -> str:
    """
    Send a prompt to Ollama and get a response.

    Args:
        prompt: the full prompt (usually context + question)
        model: Ollama model name
        system_prompt: optional system instruction
        base_url: Ollama server URL
        timeout: request timeout in seconds

    Returns:
        model's response as string
    """
    if not is_ollama_running(base_url):
        raise ConnectionError(
            "Ollama is not running.\n"
            "Start it with: ollama serve\n"
            "Then check available models with: ollama list"
        )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
    }

    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"].strip()

    except requests.Timeout:
        raise TimeoutError(f"Ollama timed out after {timeout}s. Try a smaller model.")
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


# ── RAG Answer Generation ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are FarrahAI, a subject-specific exam assistant.
You ONLY answer based on the provided notes context.
If the answer is not in the context, say: "This topic isn't covered in the uploaded notes."
Be concise, clear, and exam-focused.
Do not add information from outside the provided context."""


def answer_from_notes(query: str,
                       context: str,
                       model: str = DEFAULT_MODEL,
                       base_url: str = OLLAMA_BASE_URL) -> str:
    """
    Generate an answer to the student's query using retrieved note context.

    Args:
        query: student's question
        context: formatted retrieved note chunks (from retriever.py)
        model: Ollama model

    Returns:
        answer string
    """
    prompt = f"""Based on the following notes, answer the question.

=== NOTES CONTEXT ===
{context}

=== QUESTION ===
{query}

=== ANSWER ==="""

    logger.info(f"Sending to Ollama [{model}]: {query[:60]}...")
    answer = chat(prompt, model=model, system_prompt=SYSTEM_PROMPT, base_url=base_url)
    return answer


def explain_prediction(predicted_topics: list[dict],
                        teacher_name: str,
                        model: str = DEFAULT_MODEL,
                        base_url: str = OLLAMA_BASE_URL) -> str:
    """
    Ask Ollama to explain the topic prediction in a student-friendly way.
    """
    topics_str = "\n".join(
        [f"  {t['rank']}. {t['topic']} (appeared {t['historical_count']} times)"
         for t in predicted_topics[:8]]
    )

    prompt = f"""Based on {teacher_name}'s historical question papers, these are the top predicted topics:

{topics_str}

Write a concise 3-4 line study recommendation for students preparing for the exam.
Focus on what to prioritize and why."""

    return chat(prompt, model=model, base_url=base_url)
