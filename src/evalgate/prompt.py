"""One prompt format, shared by training and inference so they can never drift apart."""

from __future__ import annotations

import re

SYSTEM = ("You translate questions into a single SQLite query. Use only the tables and columns in the schema. "
          "Reply with the SQL query only, no explanation.")


def messages(question: str, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Schema:\n{context}\n\nQuestion: {question}"},
    ]


def extract_sql(text: str) -> str:
    """Pull the query out of a model reply: drop code fences and anything after the first statement."""
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    text = text.strip()
    start = re.search(r"\b(select|with)\b", text, re.IGNORECASE)
    if start:
        text = text[start.start():]
    return text.split(";")[0].strip()
