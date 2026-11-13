from evalgate.metrics import categories
from evalgate.prompt import extract_sql


def test_extract_sql_from_fenced_reply():
    assert extract_sql("Here you go:\n```sql\nSELECT a FROM t;\n```\nThis selects a.") == "SELECT a FROM t"


def test_extract_sql_drops_preamble_and_trailing_statements():
    assert extract_sql("The query is: SELECT a FROM t; SELECT b FROM t") == "SELECT a FROM t"


def test_categories():
    assert categories("SELECT COUNT(*) FROM t GROUP BY x") == ["aggregate", "group_by"]
    assert categories("SELECT a FROM t WHERE b = 1") == ["simple_filter"]
    assert "nested" in categories("SELECT a FROM t WHERE b IN (SELECT b FROM u)")
