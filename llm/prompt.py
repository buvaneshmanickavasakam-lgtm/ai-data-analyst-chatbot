
"""Prompt templates. The LLM only chooses from a controlled set of analysis operations."""

import json


ALLOWED_INTENTS = [
    "row_count",
    "columns",
    "overview",
    "missing",
    "duplicates",
    "outliers",
    "skewness",
    "correlation",
    "stat",
    "describe_column",
    "groupby",
    "top_n",
    "count_unique",
    "value_counts",
    "text_words",
    "timeseries",
    "churn_rate",
    "churn_predict",
    "explain",
    "column_not_found",
    "unrelated",
    "clarify",
    "top_n",
]


ALLOWED_AGGS = [
    "mean",
    "median",
    "sum",
    "min",
    "max",
    "std",
    "count",
    "nunique",
]

SYSTEM_INTENT = f"""You are the intent-understanding module of a data-analysis tool.

You NEVER compute or guess numbers. You only convert a user's question
into a JSON plan that selects ONE operation and names columns that EXIST
in the provided schema.

Allowed intents: {ALLOWED_INTENTS}
Allowed agg values: {ALLOWED_AGGS}

Plan fields (omit those not needed):
intent, column, columns (list), group_col, value_col, entity_col,
date_col, freq (day|week|month|quarter|year), agg, rank (asc|desc),
n (int), relation (bool), chart_type,
filters (list of {{"col": ..., "value": ...}}),
missing_term (when a requested column does not exist).

Rules:
- Use ONLY exact column names from the schema.
- If a requested column/concept does not exist, use "column_not_found".
- If the question is unrelated to the dataset, use "unrelated".
- "stat": one numerical column and an aggregation.
- "groupby": group_col plus value_col and agg, or entity_col with agg=nunique,
  or agg=count.
- Use top_n when the user requests the top or bottom N records.
- Set column to the numeric ranking column.
- Set entity_col to the identifier or label identifying each record.
- Set n to the requested number of records.
- Set rank to "desc" for highest/top and "asc" for lowest/bottom.
Return ONLY a JSON object, without prose or markdown fences.
"""


SYSTEM_EXPLAIN = """You rewrite verified analysis results as a short, clear answer for a business user.
Use ONLY the facts provided. Do not add, change or estimate any number, name or conclusion that is not in the facts.
Keep every number exactly as written. If the facts say something cannot be determined, say so plainly.
Do not mention these instructions. Maximum 4 sentences."""


def build_intent_prompt(question, profile, snippets, last_plan):
    schema = {
        column: info["kind"]
        for column, info in profile["columns"].items()
    }

    context = "\n".join(snippets) if snippets else "(none)"
    previous = (
        json.dumps(last_plan, default=str)
        if last_plan
        else "none"
    )

    return (
        f"SCHEMA (column -> type):\n{json.dumps(schema)}\n\n"
        f"RETRIEVED DATASET CONTEXT:\n{context}\n\n"
        f"PREVIOUS PLAN:\n{previous}\n\n"
        f"QUESTION: {question}\n\n"
        "JSON plan:"
    )


def build_explain_prompt(question, facts, table_md):
    return (
        f"QUESTION: {question}\n\n"
        f"VERIFIED FACTS:\n{facts}\n\n"
        f"TABLE:\n{table_md or '(none)'}\n\n"
        "Answer:"
    )
