"""History management module for conversation compression.

Follows Single Responsibility Principle - handles only history-related operations.
"""

from __future__ import annotations

from donkit.llm import GenerateRequest, LLMModelAbstract, Message
from loguru import logger

HISTORY_COMPRESSION_THRESHOLD = 5  # Compress when user messages exceed this
HISTORY_KEEP_RECENT = 1  # Keep last N messages after compression

HISTORY_SUMMARY_PROMPT = """Summarize this conversation concisely.
Preserve ALL key information: file paths, project names, configurations, decisions, errors.
Format as bullet points. Be brief but complete."""


async def compress_history_if_needed(
    history: list[Message],
    provider: LLMModelAbstract,
) -> list[Message]:
    """Compress history when it exceeds threshold by generating a summary.

    Args:
        history: List of conversation messages
        provider: LLM provider for generating summary

    Returns:
        Compressed history list or original if no compression needed
    """
    user_msg_count = sum(1 for m in history if m.role == "user")
    if user_msg_count <= HISTORY_COMPRESSION_THRESHOLD:
        return history

    # Separate system messages and conversation
    system_msgs = [m for m in history if m.role == "system"]
    conversation_msgs = [m for m in history if m.role != "system"]

    # Keep last N messages
    msgs_to_summarize = conversation_msgs[:-HISTORY_KEEP_RECENT]
    msgs_to_keep = conversation_msgs[-HISTORY_KEEP_RECENT:]

    if not msgs_to_summarize:
        return history

    # Generate summary using LLM - pass conversation as messages
    try:
        request = GenerateRequest(
            messages=msgs_to_summarize + [Message(role="user", content=HISTORY_SUMMARY_PROMPT)]
        )
        response = await provider.generate(request)
        summary = response.content or ""
        summary_text = f"[CONVERSATION HISTORY SUMMARY]\n{summary}\n[END SUMMARY]"

        # Build new history: system + summary + recent messages
        new_history = system_msgs + [Message(role="assistant", content=summary_text)] + msgs_to_keep
        logger.info(f"Compressed history: {len(history)} -> {len(new_history)} messages")
        return new_history
    except Exception as e:
        logger.warning(f"Failed to compress history: {e}")
        return history
