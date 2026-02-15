"""LLM assistant — calls OpenRouter with RAG-enriched context."""

from typing import Optional, List, Dict
import pandas as pd

from .rag import DataRAG

SYSTEM_PROMPT = (
    "You are a helpful data analysis assistant. "
    "Answer the user's question ONLY based on the provided data context below. "
    "If the data context does not contain enough information to answer, say so. "
    "Be precise with numbers — cite exact values from the context. "
    "Use markdown formatting for clarity."
)


def _build_messages(
    question: str,
    context: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Construct the message list for the LLM."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n--- DATA CONTEXT ---\n{context}\n--- END CONTEXT ---"},
    ]
    if history:
        messages.extend(history[-10:])  # keep last 10 turns
    messages.append({"role": "user", "content": question})
    return messages


def chat_with_llm(
    api_base: str,
    api_key: str,
    model: str,
    user_message: str,
    df_context: Optional[pd.DataFrame] = None,
    history: Optional[List[Dict[str, str]]] = None,
    rag: Optional[DataRAG] = None,
    top_k: int = 8,
) -> str:
    """Send a chat message to a local LLM (Ollama) with RAG-enriched context."""
    
    # Build or reuse the RAG index
    if rag is None and df_context is not None:
        rag = DataRAG(df_context)
    
    # Retrieve relevant context
    if rag is not None:
        context = rag.build_context(user_message, top_k=top_k)
    else:
        context = "No data loaded."

    messages = _build_messages(user_message, context, history)

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=api_base,
            api_key=api_key,
        )
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content or ""
    except Exception as exc:
        return f"❌ Error connecting to Local AI: {exc}\n\nMake sure Ollama is running (`ollama serve`) and you have pulled the model (`ollama pull {model}`)."
