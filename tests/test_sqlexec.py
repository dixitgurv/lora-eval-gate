from evalgate.sqlexec import check, gold_is_usable, normalize_sql, parse_schema, query_literals

CTX = "CREATE TABLE head (age INTEGER, name VARCHAR, born_state VARCHAR)"


def test_parse_schema_multiple_tables():
    tables = parse_schema("CREATE TABLE a (x INTEGER, y VARCHAR), CREATE TABLE b (x INTEGER, z REAL)")
    assert [t.name for t in tables] == ["a", "b"]
    assert [(c.name, c.numeric) for c in tables[1].columns] == [("x", True), ("z", True)]
    assert tables[0].columns[1].numeric is False


def test_literals_include_double_quoted_values():
    strings, numbers = query_literals('SELECT name FROM head WHERE born_state = "California" AND age > 56')
    assert "California" in strings
    assert 56.0 in numbers


def test_equivalent_queries_match_by_execution():
    gold = "SELECT COUNT(*) FROM head WHERE age > 5"
    res = check(CTX, gold, "select count(*) from head where 5 < age")
    assert res.correct and res.method == "execution"


def test_wrong_filter_is_caught():
    res = check(CTX, "SELECT COUNT(*) FROM head WHERE age > 5", "SELECT COUNT(*) FROM head WHERE age < 5")
    assert not res.correct and res.valid


def test_invalid_sql_is_counted_invalid():
    res = check(CTX, "SELECT name FROM head", "SELEC name FRM head")
    assert not res.valid and not res.correct


def test_unknown_column_is_invalid():
    res = check(CTX, "SELECT name FROM head", "SELECT nickname FROM head")
    assert not res.valid


def test_order_matters_only_when_gold_orders():
    gold = "SELECT name FROM head ORDER BY age"
    assert not check(CTX, gold, "SELECT name FROM head ORDER BY age DESC").correct
    assert check(CTX, "SELECT name FROM head", "SELECT name FROM head ORDER BY age DESC").correct


def test_gold_value_from_question_is_present_in_data():
    gold = 'SELECT name FROM head WHERE born_state = "California"'
    assert gold_is_usable(CTX, gold)
    assert check(CTX, gold, "SELECT name FROM head WHERE born_state = 'California'").correct
    assert not check(CTX, gold, "SELECT name FROM head WHERE born_state = 'Texas'").correct


def test_empty_gold_falls_back_to_exact_match():
    gold = "SELECT name FROM head WHERE age > 5 AND age < 5"  # can never return rows
    res = check(CTX, gold, "SELECT name FROM head WHERE age > 6 AND age < 6")
    assert res.method == "exact_fallback" and not res.correct  # both empty, but not the same query
    assert check(CTX, gold, "select name from head where age > 5 and age < 5").correct


def test_literal_neighbours_make_threshold_filters_bite():
    gold = "SELECT name FROM head WHERE age > 1000"
    assert gold_is_usable(CTX, gold)
    assert not check(CTX, gold, "SELECT name FROM head WHERE age >= 1000").correct


def test_normalize_sql():
    assert normalize_sql("SELECT  a FROM t WHERE b = \"x\";") == normalize_sql("select a from t where b = 'x'")


def test_duplicate_columns_and_tables_in_schema():
    tables = parse_schema("CREATE TABLE r (classroom VARCHAR, classroom VARCHAR), CREATE TABLE r (x INTEGER)")
    assert len(tables) == 1 and [c.name for c in tables[0].columns] == ["classroom"]


def test_normalize_keeps_case_of_string_values():
    assert normalize_sql("SELECT a FROM t WHERE c = 'Silverstone'") != normalize_sql('select a from t where c = "silverstone"')
    assert normalize_sql("SELECT a FROM T WHERE c = 'X'") == normalize_sql('select A from t where C = "X"')
