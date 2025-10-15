from typing import Optional, List, Dict

import pandas as pd

try:
    # openai>=1.0 style client (works with OpenRouter compatible endpoints)
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


SYSTEM_PROMPT = (
    "You are a helpful data assistant. Answer in simple, concise language. "
    "If the user asks about the uploaded data, summarize, describe columns, suggest charts, "
    "and propose modeling steps. If you don't know, say so briefly."
)


def dataframe_brief(df: pd.DataFrame, max_rows: int = 5) -> str:
    """Return a brief text summary of a dataframe for grounding the model."""

    lines = [
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        f"Columns: {', '.join(df.columns[:20])}{'...' if len(df.columns) > 20 else ''}",
        "Head:",
        df.head(max_rows).to_csv(index=False),
    ]
    return "\n".join(lines)


def chat_with_openrouter(
    api_key: Optional[str],
    model: str,
    user_message: str,
    df_context: Optional[pd.DataFrame] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Send a chat message via OpenRouter-compatible API. Returns assistant reply.

    Requires OPENROUTER_API_KEY set in settings. If unavailable, returns a helpful message.
    """

    if not api_key or OpenAI is None:
        return (
            "Chat assistant unavailable. Set ADV_OPENROUTER_API_KEY in your environment and install the 'openai' package "
            "(pip install openai) to enable chat."
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Avoid 'system'/'developer' roles for providers that don't support them (e.g., some gemma/gemini backends)
    messages: List[Dict[str, str]] = []
    preface = SYSTEM_PROMPT
    if df_context is not None:
        preface += "\n\nData context:\n" + dataframe_brief(df_context)
    messages.append({"role": "user", "content": preface})
    if history:
        # Only keep user/assistant roles to maximize provider compatibility
        for m in history:
            if m.get("role") in {"user", "assistant"}:
                messages.append({"role": m["role"], "content": m.get("content", "")})
    messages.append({"role": "user", "content": user_message})

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=600,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:  # return provider error text to UI
        return f"Chat error: {getattr(exc, 'message', str(exc))}"


