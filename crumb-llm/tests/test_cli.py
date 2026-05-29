import json

from crumb_llm import cli


def test_help_runs(capsys):
    try:
        cli.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "CrumbLLM" in out


def test_analyze_default_provider(basic_crumb, capsys):
    code = cli.main(["analyze", str(basic_crumb)])
    assert code == 0
    out = capsys.readouterr().out
    assert out.strip()


def test_analyze_pack(example_pack, capsys):
    code = cli.main(["analyze-pack", str(example_pack)])
    assert code == 0


def test_risks_json_format(basic_crumb, capsys):
    code = cli.main(["risks", str(basic_crumb), "--format", "json"])
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["task"] == "risks"
    assert "warnings" in payload


def test_summarize_to_file(basic_crumb, tmp_path):
    out_file = tmp_path / "summary.txt"
    code = cli.main(["summarize", str(basic_crumb), "--out", str(out_file)])
    assert code == 0
    assert out_file.exists() and out_file.read_text().strip()


def test_status(capsys):
    code = cli.main(["status"])
    assert code == 0
    info = json.loads(capsys.readouterr().out)
    assert "config_path" in info


def test_setup_writes_no_secret(tmp_path, monkeypatch, capsys):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("CRUMB_LLM_CONFIG", str(cfg_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-written")
    code = cli.main(["setup", "openai", "--model", "gpt-4o-mini"])
    assert code == 0
    saved = json.loads(cfg_path.read_text())
    assert saved["provider"] == "openai"
    assert saved["model"] == "gpt-4o-mini"
    # The secret must never appear anywhere in the config file.
    assert "sk-should-not-be-written" not in cfg_path.read_text()


def test_export_dataset(example_pack, tmp_path, capsys):
    out_file = tmp_path / "data" / "crumb_training.jsonl"
    code = cli.main(["export-dataset", str(example_pack), "--out", str(out_file)])
    assert code == 0
    assert out_file.exists()
    lines = [l for l in out_file.read_text().splitlines() if l.strip()]
    assert lines
    # Each line is valid JSON.
    for line in lines:
        json.loads(line)
