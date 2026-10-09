"""Thin wrapper around the Anthropic API (optional)."""
import os

DEFAULT_MODEL = os.environ.get("AI_ANALYST_MODEL", "claude-sonnet-5-5")


def available():
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def complete(system, user, api_key, model=None, max_tokens=700):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(model=model or DEFAULT_MODEL, max_tokens=max_tokens, temperature=0,
                                 system=system, messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
