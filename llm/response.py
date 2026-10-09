"""Optional LLM wording of verified results, guarded so it cannot introduce new numbers."""
import re
from . import client
from .prompt import SYSTEM_EXPLAIN, build_explain_prompt

_NUM = re.compile(r"\d[\d,]*\.?\d*")


def _nums(text):
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(text)}


def explain(question, result, llm_cfg=None):
    """Return the answer text. Without an LLM (or if the LLM output is not grounded) the engine's own text is used."""
    facts = result["text"]
    if not (result.get("ok") and llm_cfg and llm_cfg.get("api_key") and client.available()):
        return facts
    table_md = ""
    t = result.get("table")
    if t is not None and len(t):
        table_md = t.head(15).to_csv(index=False)
    allowed = _nums(facts + " " + table_md) | {str(i) for i in range(0, 11)}
    try:
        out = client.complete(SYSTEM_EXPLAIN, build_explain_prompt(question, facts, table_md),
                              llm_cfg["api_key"], llm_cfg.get("model"), 400).strip()
        if out and _nums(out) <= allowed:
            return out
    except Exception:
        pass
    return facts
