from crumb_llm.crumb.loader import load_crumb
from crumb_llm.crumb.pack_reader import read_pack
from crumb_llm.crumb.parser_adapter import validate_text


def test_load_crumb(basic_crumb):
    doc = load_crumb(basic_crumb)
    assert doc.kind == "task"
    assert "goal" in doc.sections
    # File paths referenced in the CRUMB are extracted for grounding.
    paths = doc.file_paths()
    assert "src/gateway/middleware.py" in paths
    assert "config/redis.yaml" in paths


def test_read_pack(example_pack):
    pack = read_pack(example_pack)
    assert len(pack) == 2
    assert pack.manifest is not None
    assert "src/checkout/tax.py" in pack.file_paths()


def test_validate_text_good(basic_crumb):
    text = basic_crumb.read_text()
    assert validate_text(text) == []


def test_validate_text_bad():
    errors = validate_text("not a crumb file")
    assert errors  # crumb-format reports the structural problem
