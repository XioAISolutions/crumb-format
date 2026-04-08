"""Tests for CRUMB protocol upgrades: pack, bridges, lint, and fixtures."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))
import crumb


TASK_CRUMB = """\
BEGIN CRUMB
v=1.1
kind=task
title=Fix auth redirect
source=cursor.agent
project=auth-app
---
[goal]
Fix the auth redirect loop after refresh.

[context]
- JWT cookie auth
- Redirect loop happens on full page refresh

[constraints]
- Keep the login UI unchanged
- Preserve current cookie names
END CRUMB
"""

MEM_CRUMB = """\
BEGIN CRUMB
v=1.1
kind=mem
title=Team memory
source=human.notes
project=auth-app
---
[consolidated]
- Prefers minimal auth changes
- Always preserve backwards compatibility for cookies
END CRUMB
"""

MAP_CRUMB = """\
BEGIN CRUMB
v=1.1
kind=map
title=Auth project map
source=human.notes
project=auth-app
---
[project]
Next.js auth application with middleware-based refresh handling.

[modules]
- src/auth.ts
- src/middleware.ts
- tests/auth-refresh.spec.ts
END CRUMB
"""

TODO_CRUMB = """\
BEGIN CRUMB
v=1.1
kind=todo
title=Auth todo
source=cli
project=auth-app
---
[tasks]
- [ ] Verify refresh redirect behavior
- [ ] Add auth regression test
END CRUMB
"""


def _init_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True, capture_output=True, text=True)


def _fixture_root() -> Path:
    return Path(__file__).resolve().parent.parent / "fixtures"


class TestPackCommand:
    def test_pack_builds_valid_task_handoff(self, tmp_path):
        crumbs_dir = tmp_path / "crumbs"
        crumbs_dir.mkdir()
        (crumbs_dir / "task.crumb").write_text(TASK_CRUMB, encoding="utf-8")
        (crumbs_dir / "mem.crumb").write_text(MEM_CRUMB, encoding="utf-8")
        (crumbs_dir / "map.crumb").write_text(MAP_CRUMB, encoding="utf-8")
        (crumbs_dir / "todo.crumb").write_text(TODO_CRUMB, encoding="utf-8")

        (crumbs_dir / "src").mkdir()
        (crumbs_dir / "src" / "auth.ts").write_text("export const auth = true;\n", encoding="utf-8")
        (crumbs_dir / "src" / "middleware.ts").write_text("export function middleware() {}\n", encoding="utf-8")
        _init_repo(crumbs_dir)
        subprocess.run(["git", "add", "."], cwd=crumbs_dir, check=True, capture_output=True, text=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=crumbs_dir, check=True, capture_output=True, text=True)
        (crumbs_dir / "src" / "auth.ts").write_text(
            "export const auth = true;\nexport function refreshRedirect() {}\n",
            encoding="utf-8",
        )

        output = tmp_path / "handoff.crumb"
        crumb.main(
            [
                "pack",
                "--dir",
                str(crumbs_dir),
                "--query",
                "auth redirect refresh",
                "--kind",
                "task",
                "--max-total-tokens",
                "1800",
                "-o",
                str(output),
            ]
        )
        rendered = output.read_text(encoding="utf-8")
        parsed = crumb.parse_crumb(rendered)
        assert parsed["headers"]["kind"] == "task"
        assert parsed["headers"]["source"] == "crumb.pack"
        assert parsed["headers"]["extensions"] == "crumb.pack.v1"
        assert parsed["sections"]["goal"]
        assert parsed["sections"]["context"]
        assert parsed["sections"]["constraints"]
        assert "auth" in rendered.lower()
        assert crumb.estimate_tokens(rendered) <= 1800

    def test_pack_filters_irrelevant_context_and_dedupes_constraints(self, tmp_path):
        crumbs_dir = tmp_path / "crumbs"
        crumbs_dir.mkdir()
        (crumbs_dir / "task-auth.crumb").write_text(TASK_CRUMB, encoding="utf-8")
        (crumbs_dir / "mem-auth.crumb").write_text(MEM_CRUMB, encoding="utf-8")
        (crumbs_dir / "map-auth.crumb").write_text(MAP_CRUMB, encoding="utf-8")
        (crumbs_dir / "other.crumb").write_text(
            """\
BEGIN CRUMB
v=1.1
kind=task
title=Luxury builder homepage
source=human.notes
---
[goal]
Improve authority and premium trust signals.

[context]
- Luxury custom home builder website focused on premium trust.

[constraints]
- Keep copy concise and sales-forward.
END CRUMB
""",
            encoding="utf-8",
        )

        output = tmp_path / "handoff.crumb"
        crumb.main(
            [
                "pack",
                "--dir",
                str(crumbs_dir),
                "--query",
                "auth redirect refresh",
                "--kind",
                "task",
                "--max-total-tokens",
                "1800",
                "--strategy",
                "hybrid",
                "-o",
                str(output),
            ]
        )
        rendered = output.read_text(encoding="utf-8")
        assert "Luxury custom home builder" not in rendered
        constraints = crumb.parse_crumb(rendered)["sections"]["constraints"]
        normalized = [crumb.normalize_entry(line) for line in constraints if line.strip()]
        assert len(normalized) == len(set(normalized))
        assert sum(1 for item in normalized if "login ui" in item) == 1

    def test_pack_mode_debug_prefers_log_evidence(self, tmp_path):
        crumbs_dir = tmp_path / "crumbs"
        crumbs_dir.mkdir()
        (crumbs_dir / "task-auth.crumb").write_text(TASK_CRUMB, encoding="utf-8")
        (crumbs_dir / "mem-auth.crumb").write_text(MEM_CRUMB, encoding="utf-8")
        (crumbs_dir / "log-auth.crumb").write_text(
            """\
BEGIN CRUMB
v=1.1
kind=log
title=Auth incident log
source=cli
---
[entries]
- [2026-04-08T15:00:00Z] Refresh redirect reproduced after hard reload.
- [2026-04-08T15:02:00Z] Middleware throws before cookie parsing settles.
END CRUMB
""",
            encoding="utf-8",
        )

        output = tmp_path / "debug.crumb"
        crumb.main(
            [
                "pack",
                "--dir",
                str(crumbs_dir),
                "--query",
                "auth redirect refresh middleware",
                "--kind",
                "task",
                "--mode",
                "debug",
                "--max-total-tokens",
                "1800",
                "-o",
                str(output),
            ]
        )
        parsed = crumb.parse_crumb(output.read_text(encoding="utf-8"))
        rendered = output.read_text(encoding="utf-8")
        assert parsed["headers"]["x-crumb-pack.mode"] == "debug"
        assert parsed["sections"]["goal"][0].startswith("Diagnose ")
        assert parsed["sections"]["context"][0].startswith("- Observed symptom:")
        assert any("Evidence:" in line or "Observed symptom:" in line for line in parsed["sections"]["context"])
        assert any("Refresh redirect reproduced" in line for line in parsed["sections"]["context"])
        assert not any(line.startswith("- Likely cause:") for line in parsed["sections"]["constraints"])
        assert "log-auth.crumb" in rendered

    def test_pack_mode_review_relabels_git_scope(self, tmp_path):
        crumbs_dir = tmp_path / "crumbs"
        crumbs_dir.mkdir()
        (crumbs_dir / "task-auth.crumb").write_text(TASK_CRUMB, encoding="utf-8")
        (crumbs_dir / "mem-auth.crumb").write_text(MEM_CRUMB, encoding="utf-8")
        (crumbs_dir / "map-auth.crumb").write_text(MAP_CRUMB, encoding="utf-8")
        (crumbs_dir / "src").mkdir()
        (crumbs_dir / "src" / "auth.ts").write_text("export const auth = true;\n", encoding="utf-8")
        _init_repo(crumbs_dir)
        subprocess.run(["git", "add", "."], cwd=crumbs_dir, check=True, capture_output=True, text=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=crumbs_dir, check=True, capture_output=True, text=True)
        (crumbs_dir / "src" / "auth.ts").write_text("export const auth = true;\nexport const redirect = true;\n", encoding="utf-8")

        output = tmp_path / "review.crumb"
        crumb.main(
            [
                "pack",
                "--dir",
                str(crumbs_dir),
                "--query",
                "auth redirect",
                "--kind",
                "task",
                "--mode",
                "review",
                "--max-total-tokens",
                "1800",
                "-o",
                str(output),
            ]
        )
        parsed = crumb.parse_crumb(output.read_text(encoding="utf-8"))
        assert parsed["headers"]["x-crumb-pack.mode"] == "review"
        assert parsed["sections"]["goal"][0].startswith("Review the work scoped in this pack")
        assert parsed["sections"]["context"][0].startswith("- Review scope:")
        assert any("Affected module:" in line for line in parsed["sections"]["context"])
        assert any("Current invariant:" in line for line in parsed["sections"]["context"])
        assert not any(line.startswith("- Implementation detail:") for line in parsed["sections"]["constraints"])

    def test_pack_mode_implement_shapes_context_for_execution(self, tmp_path):
        crumbs_dir = tmp_path / "crumbs"
        crumbs_dir.mkdir()
        (crumbs_dir / "task-auth.crumb").write_text(TASK_CRUMB, encoding="utf-8")
        (crumbs_dir / "mem-auth.crumb").write_text(MEM_CRUMB, encoding="utf-8")
        (crumbs_dir / "map-auth.crumb").write_text(MAP_CRUMB, encoding="utf-8")
        (crumbs_dir / "todo-auth.crumb").write_text(TODO_CRUMB, encoding="utf-8")

        output = tmp_path / "implement.crumb"
        crumb.main(
            [
                "pack",
                "--dir",
                str(crumbs_dir),
                "--query",
                "auth redirect refresh middleware",
                "--kind",
                "task",
                "--mode",
                "implement",
                "--max-total-tokens",
                "1800",
                "-o",
                str(output),
            ]
        )
        parsed = crumb.parse_crumb(output.read_text(encoding="utf-8"))
        assert parsed["headers"]["x-crumb-pack.mode"] == "implement"
        assert parsed["sections"]["goal"][0].startswith("Fix ")
        assert parsed["sections"]["context"][0].startswith("- Start in:")
        assert any("Current baseline:" in line for line in parsed["sections"]["context"])
        assert any("Next step:" in line for line in parsed["sections"]["context"])
        assert not any(line.startswith("- Implementation detail:") for line in parsed["sections"]["constraints"])


class TestLintCommand:
    def test_lint_flags_secrets_with_strict_exit(self, tmp_path, capsys):
        suspect = tmp_path / "handoff.crumb"
        suspect.write_text(
            TASK_CRUMB.replace("- JWT cookie auth", "- api_key=sk-1234567890abcdefghijklmnopqrst"),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as exc:
            crumb.main(["lint", str(suspect), "--secrets", "--strict"])
        assert exc.value.code == 1
        combined = capsys.readouterr()
        assert "openai_key" in combined.err or "openai_key" in combined.out

    def test_lint_redacts_in_place(self, tmp_path):
        suspect = tmp_path / "handoff.crumb"
        suspect.write_text(
            TASK_CRUMB.replace("- JWT cookie auth", "- token=ghp_1234567890abcdefghijklmnop"),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as exc:
            crumb.main(["lint", str(suspect), "--secrets", "--redact"])
        assert exc.value.code == 1
        text = suspect.read_text(encoding="utf-8")
        assert "[REDACTED:github_token]" in text


class TestMempalaceBridge:
    def test_bridge_export_from_input_generates_valid_crumb(self, tmp_path):
        export_text = tmp_path / "mempalace.txt"
        export_text.write_text(
            "Auth migration moved refresh logic into middleware.ts.\nOld session cookies still validate.\n",
            encoding="utf-8",
        )
        output_dir = tmp_path / "out"
        crumb.main(
            [
                "bridge",
                "mempalace",
                "export",
                "--input",
                str(export_text),
                "--query",
                "auth migration",
                "--as",
                "task",
                "-o",
                str(output_dir),
            ]
        )
        generated = list(output_dir.glob("*.crumb"))
        assert len(generated) == 1
        parsed = crumb.parse_crumb(generated[0].read_text(encoding="utf-8"))
        assert parsed["headers"]["source"] == "mempalace.bridge"
        assert parsed["headers"]["extensions"] == "bridge.mempalace.export.v1"
        assert parsed["headers"]["kind"] == "task"

    def test_bridge_export_fails_cleanly_when_backend_unavailable(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(SystemExit) as exc:
            crumb.main(
                [
                    "bridge",
                    "mempalace",
                    "export",
                    "--query",
                    "auth migration",
                    "--as",
                    "task",
                    "-o",
                    str(tmp_path / "out"),
                ]
            )
        assert exc.value.code == 1
        assert "MemPalace CLI is not installed" in capsys.readouterr().err

    def test_bridge_import_creates_adapter_bundle(self, tmp_path):
        source = tmp_path / "task.crumb"
        source.write_text(TASK_CRUMB, encoding="utf-8")
        output = tmp_path / "bridge.json"
        crumb.main(["bridge", "mempalace", "import", str(source), "-o", str(output), "--wing", "work"])
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["adapter"] == "mempalace"
        assert payload["supported"]["direct_write"] is False
        assert payload["records"][0]["wing"] == "work"
        assert payload["records"][0]["kind"] == "task"


class TestMCPProtocolTools:
    def test_mcp_pack_tool(self, tmp_path):
        crumbs_dir = tmp_path / "crumbs"
        crumbs_dir.mkdir()
        (crumbs_dir / "task.crumb").write_text(TASK_CRUMB, encoding="utf-8")
        (crumbs_dir / "mem.crumb").write_text(MEM_CRUMB, encoding="utf-8")
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp"))
        import server as mcp_server

        result = mcp_server.handle_tool_call(
            "crumb_pack",
            {
                "dir": str(crumbs_dir),
                "query": "auth redirect",
                "kind": "task",
                "max_total_tokens": 1200,
                "strategy": "hybrid",
            },
        )
        assert "BEGIN CRUMB" in result
        assert "source=crumb.pack" in result

    def test_mcp_lint_tool(self, tmp_path):
        target = tmp_path / "handoff.crumb"
        target.write_text(
            TASK_CRUMB.replace("- JWT cookie auth", "- api_key=sk-1234567890abcdefghijklmnopqrst"),
            encoding="utf-8",
        )
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp"))
        import server as mcp_server

        result = mcp_server.handle_tool_call(
            "crumb_lint",
            {"files": [str(target)], "secrets": True, "strict": True},
        )
        assert "openai_key" in result


class TestFixtureSuite:
    @pytest.mark.parametrize(
        "fixture_path",
        sorted((_fixture_root() / "valid").glob("*.crumb")) + sorted((_fixture_root() / "extensions").glob("*.crumb")),
    )
    def test_valid_and_extension_fixtures_match_expected_json(self, fixture_path):
        expected = json.loads(fixture_path.with_suffix(".expected.json").read_text(encoding="utf-8"))
        parsed = crumb.parse_crumb(fixture_path.read_text(encoding="utf-8"))
        assert parsed == expected

    @pytest.mark.parametrize("fixture_path", sorted((_fixture_root() / "invalid").glob("*.crumb")))
    def test_invalid_fixtures_report_expected_error(self, fixture_path):
        expected = fixture_path.with_suffix(".expected.txt").read_text(encoding="utf-8").strip()
        with pytest.raises(ValueError) as exc:
            crumb.parse_crumb(fixture_path.read_text(encoding="utf-8"))
        assert expected in str(exc.value)

    def test_cli_validate_supports_globs(self):
        pattern = str(_fixture_root() / "valid" / "*.crumb")
        with pytest.raises(SystemExit) as exc:
            crumb.main(["validate", pattern])
        assert exc.value.code == 0

    def test_python_validator_uses_fixture_suite(self):
        root = _fixture_root()
        valid = subprocess.run(
            [sys.executable, "validators/validate.py", str(root / "valid"), str(root / "extensions")],
            cwd=Path(__file__).resolve().parent.parent,
            check=False,
            capture_output=True,
            text=True,
        )
        invalid = subprocess.run(
            [sys.executable, "validators/validate.py", str(root / "invalid")],
            cwd=Path(__file__).resolve().parent.parent,
            check=False,
            capture_output=True,
            text=True,
        )
        assert valid.returncode == 0, valid.stderr
        assert invalid.returncode == 1

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    def test_node_validator_uses_fixture_suite(self):
        root = _fixture_root()
        valid = subprocess.run(
            ["node", "validators/validate.js", str(root / "valid"), str(root / "extensions")],
            cwd=Path(__file__).resolve().parent.parent,
            check=False,
            capture_output=True,
            text=True,
        )
        invalid = subprocess.run(
            ["node", "validators/validate.js", str(root / "invalid")],
            cwd=Path(__file__).resolve().parent.parent,
            check=False,
            capture_output=True,
            text=True,
        )
        assert valid.returncode == 0, valid.stderr
        assert invalid.returncode == 1
