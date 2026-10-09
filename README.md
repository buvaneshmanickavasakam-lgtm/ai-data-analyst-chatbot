# 📊 AI Data Analyst Chatbot

A Streamlit app that profiles one uploaded dataset (CSV / Excel) and answers natural-language questions about it.
**All numbers come from Pandas/NumPy — never from an LLM.** If something can't be determined from the data, the app says so.

## Run
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python make_sample_data.py                            # optional: creates demo files in Datasets/
streamlit run app.py
```
Upload a file in the sidebar, then chat. Type **chart** to visualise the previous answer, **exit** to end the session.

## Example questions
- What is the average salary? → *Which department has the highest?* → *chart*
- Which columns contain missing values? · Find the outliers in age · Correlation between age and salary
- Show monthly sales · Which country has the most customers? · What is the churn rate? · Top 5 products by revenue
- average salary in IT (filters by a category value) · *why?* (factual breakdown of the previous grouped result)

## Architecture
```
Question → intent (rules, or optional Claude → JSON plan) → plan validated against real columns/types
        → Pandas analysis (analysis/query_engine.py) → result validation → answer text
        → optional Plotly chart (visualization/decision_engine.py)
```
| Package | Role |
|---|---|
| `analysis/` | loading, type detection (numerical/categorical/datetime/boolean/text/identifier), missing, duplicates, stats, cardinality, IQR outliers, skewness, correlation, controlled query engine |
| `visualization/` | decision engine + Plotly charts (bar, line, histogram, box, scatter, adaptive correlation heatmap) |
| `llm/` | rule-based parser, optional Claude parser/explainer; plans are validated, explanations are rejected if they contain numbers not in the verified facts |
| `rag/` | profile → documents → chunks → ChromaDB (in-memory; NumPy fallback) → retrieval of schema context. Never used for calculation |

### No-hallucination safeguards
- The LLM can only choose from a fixed list of operations and **existing** column names; invalid plans fall back to the rule parser.
- No LLM-generated code is ever executed.
- Standard refusals: *"I cannot find a column related to 'x'…"*, *"I can only answer questions based on the currently uploaded dataset."*, *"I cannot determine this reliably…"*, *"I cannot generate a meaningful chart…"*.
- Type checks (no mean of a text/ID column), minimum valid-record checks, missing values reported.
- A derived `Revenue = Quantity × Price` column is added only if no revenue/sales column exists, and the app tells you.

## Optional LLM
`pip install anthropic`, tick **Use Claude** in the sidebar and paste a key (or set `ANTHROPIC_API_KEY`). Model: `AI_ANALYST_MODEL` env var (default `claude-sonnet-5-5`).
Better embeddings: `pip install sentence-transformers` (otherwise a built-in hashing embedder is used).

## Notes / limits
- Individual-level predictions (e.g. "which customers will churn") are deliberately declined; no model is trained.
- Parsing is heuristic when no LLM is enabled: use column names or close words from your data. Enable the LLM for looser phrasing.
- Charts sample large data (scatter 5k points; histogram 100k) to stay responsive.
