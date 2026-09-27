"""
ai_assistant.py
-----------------
A real-AI chatbot embedded in the app. It can answer questions about the
uploaded image, the currently applied point operation and its numbers,
or general Digital Image Processing questions — using the user's own
Anthropic or OpenAI API key.

Kept separate from app.py so the UI code doesn't need to know anything
about specific provider SDKs.
"""

from __future__ import annotations

SYSTEM_INSTRUCTIONS = (
    "You are the built-in assistant inside 'Point Operation Explorer', an "
    "educational Digital Image Processing app. The user is learning about "
    "point operations (negative, log, gamma, contrast stretching, "
    "thresholding, gray-level slicing). Answer questions about their "
    "uploaded image, the operation currently applied and its numbers "
    "(given to you below), or Digital Image Processing concepts generally. "
    "Be concise and beginner-friendly, and refer to the actual numbers "
    "given below when relevant instead of speaking in generalities. If the "
    "user asks something unrelated to the app, you can still answer "
    "normally as a helpful general-purpose assistant."
)


def build_image_context(
    *,
    image_info: dict,
    original_stats: dict,
    operation_name: str | None,
    operation_meta: dict | None,
    params_used: dict | None,
    processed_stats: dict | None,
    change_summary: str | None,
) -> str:
    """
    Summarize the current app state as plain text grounding for the
    chatbot. The model never sees the actual pixels — only these
    already-computed facts — which is enough for it to answer
    image-specific questions accurately.
    """
    lines = [
        f"Current image: {image_info['filename']} "
        f"({image_info['width']}x{image_info['height']}, mode {image_info['original_mode']})",
        f"Original grayscale stats: min={original_stats['min']}, max={original_stats['max']}, "
        f"mean={original_stats['mean']}, std={original_stats['std']}, median={original_stats['median']}",
    ]
    if operation_name and operation_meta:
        lines.append(f"Operation currently applied: {operation_name}")
        lines.append(f"Formula: {operation_meta.get('formula_plain', '')}")
        if params_used:
            lines.append(f"Parameters used: {params_used}")
        if processed_stats:
            lines.append(
                f"Processed stats: min={processed_stats['min']}, max={processed_stats['max']}, "
                f"mean={processed_stats['mean']}, std={processed_stats['std']}, "
                f"median={processed_stats['median']}"
            )
        if change_summary:
            lines.append(f"Computed explanation of what changed: {change_summary}")
    else:
        lines.append("No point operation has been applied yet.")
    return "\n".join(lines)


def call_anthropic(api_key: str, model: str, system_prompt: str, history: list[dict]) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": m["role"], "content": m["content"]} for m in history],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def call_openai(api_key: str, model: str, system_prompt: str, history: list[dict]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in history
    ]
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=1000)
    return response.choices[0].message.content


def call_groq(api_key: str, model: str, system_prompt: str, history: list[dict]) -> str:
    # Groq's API is OpenAI-compatible — same client, different base_url.
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in history
    ]
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=1000)
    return response.choices[0].message.content


def get_chat_response(
    provider: str, api_key: str, model: str, context: str, history: list[dict]
) -> str:
    """
    Dispatch to the selected provider. `context` (from build_image_context)
    is appended to the fixed system instructions so every reply is grounded
    in the current image/operation.
    """
    if not api_key:
        raise ValueError(
            "No API key provided. Enter your Anthropic, OpenAI, or Groq API key in the sidebar."
        )

    system_prompt = f"{SYSTEM_INSTRUCTIONS}\n\n--- Current app state ---\n{context}"

    if provider == "Anthropic (Claude)":
        return call_anthropic(api_key, model, system_prompt, history)
    elif provider == "OpenAI (GPT)":
        return call_openai(api_key, model, system_prompt, history)
    elif provider == "Groq (Free)":
        return call_groq(api_key, model, system_prompt, history)
    raise ValueError(f"Unknown provider: {provider}")
