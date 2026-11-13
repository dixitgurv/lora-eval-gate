"""Execution-based checking of generated SQL.

Exact string match undercounts correct answers: `SELECT COUNT(*) FROM t WHERE a > 5`
and `select count(*) from t where 5 < a` are the same query. So we run both the gold
query and the prediction against small synthetic databases and compare the results.

The catch with synthetic data is false positives: two different queries can both
return an empty set. Three defences:
  * Data is built from the gold query's own literals, so WHERE clauses actually match rows.
  * Each comparison runs on several databases with different random seeds.
  * If the gold query returns nothing on every database, the execution check proves
    nothing, and we fall back to normalized exact match for that example.
"""

from __future__ import annotations

import random
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

ROWS_PER_TABLE = 30
SEEDS = (11, 23, 47)
GENERIC_STRINGS = ("alpha", "beta", "gamma")
_NUMERIC_TYPES = ("INT", "REAL", "NUM", "FLOAT", "DOUBLE", "DEC")


@dataclass(frozen=True)
class Column:
    name: str
    numeric: bool


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]


def parse_schema(context: str) -> list[Table]:
    """Parse the dataset's `CREATE TABLE t (a INTEGER, b VARCHAR), CREATE TABLE ...` context."""
    tables, seen_tables = [], set()
    for match in re.finditer(r"CREATE TABLE\s+(\w+)\s*\((.*?)\)(?=\s*,\s*CREATE TABLE|\s*$)", context, re.IGNORECASE | re.DOTALL):
        if match.group(1).lower() in seen_tables:
            continue  # the dataset occasionally repeats a CREATE TABLE
        seen_tables.add(match.group(1).lower())
        cols, seen_cols = [], set()
        for part in match.group(2).split(","):
            bits = part.strip().split()
            if not bits or bits[0].strip('"`').lower() in seen_cols:
                continue  # ...or a column
            seen_cols.add(bits[0].strip('"`').lower())
            col_type = bits[1].upper() if len(bits) > 1 else "VARCHAR"
            cols.append(Column(bits[0].strip('"`'), any(t in col_type for t in _NUMERIC_TYPES)))
        tables.append(Table(match.group(1), tuple(cols)))
    return tables


def query_literals(sql: str) -> tuple[list[str], list[float]]:
    """String and numeric literals in a query, used to seed the synthetic data."""
    strings, numbers = [], []
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.ParseError:
        tree = None
    if tree is not None:
        for lit in tree.find_all(exp.Literal):
            if lit.is_string:
                strings.append(lit.this)
            else:
                try:
                    numbers.append(float(lit.this))
                except ValueError:
                    pass
        # The dataset often writes string values in double quotes, which parse as identifiers.
        for ident in tree.find_all(exp.Identifier):
            if ident.quoted:
                strings.append(ident.this)
    strings += re.findall(r"'((?:[^']|'')*)'", sql)
    return list(dict.fromkeys(strings)), list(dict.fromkeys(numbers))


def build_db(tables: list[Table], gold_sql: str, seed: int) -> sqlite3.Connection:
    rng = random.Random(seed)
    strings, numbers = query_literals(gold_sql)
    string_pool = list(strings) + list(GENERIC_STRINGS)
    number_pool = sorted({*numbers, *(n + d for n in numbers for d in (-1, 1)), *range(10)})

    conn = sqlite3.connect(":memory:")
    for table in tables:
        cols_sql = ", ".join(f'"{c.name}" {"REAL" if c.numeric else "TEXT"}' for c in table.columns)
        conn.execute(f'CREATE TABLE "{table.name}" ({cols_sql})')
        rows = []
        for _ in range(ROWS_PER_TABLE):
            row = []
            for col in table.columns:
                if col.numeric:
                    row.append(rng.choice(number_pool))
                else:
                    # Numbers can appear as text in this dataset (e.g. "WHERE year = '2005'").
                    pool = string_pool + [str(int(n)) if float(n).is_integer() else str(n) for n in numbers]
                    row.append(rng.choice(pool))
            rows.append(row)
        placeholders = ", ".join("?" * len(table.columns))
        conn.executemany(f'INSERT INTO "{table.name}" VALUES ({placeholders})', rows)
    return conn


def _run(conn: sqlite3.Connection, sql: str) -> list[tuple]:
    conn.set_progress_handler(lambda: 1, 200_000)  # abort runaway queries (cross joins on bad output)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.set_progress_handler(None, 0)


def _canon(rows: list[tuple], ordered: bool) -> list[tuple] | Counter:
    def norm(v):
        if isinstance(v, float):
            return round(v, 4)
        if isinstance(v, str):
            try:
                return round(float(v), 4)
            except ValueError:
                return v.strip().lower()
        return v

    out = [tuple(norm(v) for v in row) for row in rows]
    return out if ordered else Counter(out)


def normalize_sql(sql: str) -> str:
    """Canonical form for exact match.

    Keywords and identifiers are case-insensitive in SQL, so they are lower-cased.
    String values are not: `'Silverstone'` and `'silverstone'` match different rows,
    so their case is kept. (An earlier version lower-cased everything, which counted
    63 of 400 baseline answers as exact matches that return the wrong rows.)
    """
    sql = sql.strip().rstrip(";")
    try:
        sql = sqlglot.transpile(sql, read="sqlite", write="sqlite")[0]
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError, IndexError):
        pass
    sql = sql.replace('"', "'")
    parts = re.split(r"('(?:[^']|'')*')", sql)  # odd indexes are string literals
    return "".join(p if i % 2 else re.sub(r"\s+", " ", p).lower() for i, p in enumerate(parts)).strip()


@dataclass
class ExecResult:
    valid: bool  # prediction parsed and ran without error
    correct: bool  # counts toward execution accuracy
    method: str  # "execution", "exact_fallback", "gold_error"
    error: str | None = None


def check(context: str, gold_sql: str, pred_sql: str) -> ExecResult:
    tables = parse_schema(context)
    ordered = bool(re.search(r"\border\s+by\b", gold_sql, re.IGNORECASE))
    exact = normalize_sql(gold_sql) == normalize_sql(pred_sql)
    gold_nonempty = False
    pred_ok = True
    first_error = None
    all_equal = True

    for seed in SEEDS:
        try:
            conn = build_db(tables, gold_sql, seed)
        except sqlite3.Error as e:
            return ExecResult(valid=True, correct=exact, method="gold_error", error=f"schema: {e}")
        try:
            gold_rows = _run(conn, gold_sql)
        except sqlite3.Error as e:
            conn.close()
            return ExecResult(valid=True, correct=exact, method="gold_error", error=f"gold: {e}")
        try:
            pred_rows = _run(conn, pred_sql)
        except (sqlite3.Error, sqlite3.Warning, ValueError) as e:
            pred_ok, first_error, all_equal = False, str(e), False
            conn.close()
            break
        conn.close()
        gold_nonempty |= bool(gold_rows) and any(any(v is not None for v in r) for r in gold_rows)
        if _canon(gold_rows, ordered) != _canon(pred_rows, ordered):
            all_equal = False

    if not pred_ok:
        return ExecResult(valid=False, correct=False, method="execution", error=first_error)
    if not gold_nonempty:
        return ExecResult(valid=True, correct=exact, method="exact_fallback")
    return ExecResult(valid=True, correct=all_equal, method="execution")


def gold_is_usable(context: str, gold_sql: str) -> bool:
    """True if the gold query runs on our synthetic databases and returns something."""
    res = check(context, gold_sql, gold_sql)
    return res.method == "execution" and res.correct
