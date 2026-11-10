from aulvero.data import acceptable_code, clean_code


def test_clean_code_preserves_indentation():
    code = "def f():\r\n    if True:\r\n        return 1  \r\n"
    cleaned = clean_code(code)
    assert "    if True:" in cleaned
    assert "        return 1" in cleaned
    assert "return 1  " not in cleaned


def test_acceptable_code_allows_symbol_heavy_source():
    code = 'def f(x):\n    return {"value": x + 1}\n' * 4
    assert acceptable_code(code, min_chars=20)
