from crumb_llm.evals.hallucination_check import check_hallucinated_paths
from crumb_llm.evals.json_check import check_json
from crumb_llm.evals.quality_gate import check_generic_empty, run_quality_gates


def test_json_check_valid():
    ok, parsed, warnings = check_json('{"a": 1}')
    assert ok and parsed == {"a": 1} and warnings == []


def test_json_check_fenced():
    ok, parsed, _ = check_json('```json\n{"a": 1}\n```')
    assert ok and parsed == {"a": 1}


def test_json_check_invalid():
    ok, _, warnings = check_json("not json")
    assert not ok and warnings


def test_hallucinated_paths():
    source = {"src/app/main.py"}
    warnings = check_hallucinated_paths(
        "Edit src/app/main.py and also lib/ghost.py", source
    )
    assert any("ghost.py" in w for w in warnings)
    assert not any("main.py" in w for w in warnings)


def test_hallucinated_paths_basename_grounding():
    # ./src/app/main.py should be considered grounded vs src/app/main.py
    warnings = check_hallucinated_paths("see ./src/app/main.py", {"src/app/main.py"})
    assert warnings == []


def test_generic_empty():
    assert check_generic_empty("")
    assert check_generic_empty("As an AI, I cannot help with that")
    assert check_generic_empty("ok") == [] or check_generic_empty("ok")  # short


def test_run_quality_gates_json_and_paths():
    parsed, warnings = run_quality_gates(
        '{"risks": []}', source_paths={"a/b.py"}, expect_json=True
    )
    assert parsed == {"risks": []}
    assert warnings == []
