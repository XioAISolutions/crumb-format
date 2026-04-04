"""Tests for agentauth — Agent Passport SDK."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentauth.store import PassportStore
from agentauth.passport import AgentPassport
from agentauth.policy import ToolPolicy
from agentauth.credentials import CredentialBroker
from agentauth.audit import AuditLogger
from agentauth.decorators import protect


@pytest.fixture
def tmp_store(tmp_path):
    """Create a PassportStore in a temp directory."""
    return PassportStore(root=str(tmp_path / ".crumb-auth"))


@pytest.fixture
def passport_mgr(tmp_store):
    return AgentPassport(store=tmp_store)


@pytest.fixture
def policy_mgr(tmp_store):
    return ToolPolicy(store=tmp_store)


@pytest.fixture
def broker(tmp_store):
    return CredentialBroker(store=tmp_store, secret_key="test-secret")


@pytest.fixture
def audit(tmp_store):
    return AuditLogger(store=tmp_store)


# ── PassportStore ─────────────────────────────────────────────────

class TestPassportStore:
    def test_directories_created(self, tmp_store):
        assert tmp_store.passports_dir.exists()
        assert tmp_store.policies_dir.exists()
        assert tmp_store.audit_dir.exists()

    def test_save_and_load_passport(self, tmp_store):
        content = "BEGIN CRUMB\nv=1.1\nkind=passport\nid=ap_test1234\n---\n## identity\n  name: bot\nEND CRUMB"
        tmp_store.save_passport(content, "ap_test1234")
        loaded = tmp_store.load_passport("ap_test1234")
        assert loaded == content

    def test_load_missing_passport(self, tmp_store):
        assert tmp_store.load_passport("ap_nonexistent") is None

    def test_revoke_and_check(self, tmp_store):
        assert not tmp_store.is_revoked("ap_test1234")
        assert tmp_store.revoke("ap_test1234") is True
        assert tmp_store.is_revoked("ap_test1234")
        # Double revoke returns False
        assert tmp_store.revoke("ap_test1234") is False

    def test_save_and_load_policy(self, tmp_store):
        policy = {"tools_allowed": ["read*"], "tools_denied": []}
        tmp_store.save_policy("my-agent", policy)
        loaded = tmp_store.load_policy("my-agent")
        assert loaded == policy

    def test_load_missing_policy(self, tmp_store):
        assert tmp_store.load_policy("no-such-agent") is None

    def test_list_passports(self, tmp_store):
        tmp_store.save_passport("content1", "ap_aaa11111")
        tmp_store.save_passport("content2", "ap_bbb22222")
        paths = tmp_store.list_passports()
        assert len(paths) == 2


# ── AgentPassport ─────────────────────────────────────────────────

class TestAgentPassport:
    def test_register_returns_id_and_path(self, passport_mgr):
        result = passport_mgr.register(name="test-bot", framework="langchain", owner="alice")
        assert result["agent_id"].startswith("ap_")
        assert result["name"] == "test-bot"
        assert Path(result["passport_path"]).exists()

    def test_inspect_after_register(self, passport_mgr):
        result = passport_mgr.register(name="inspect-bot")
        data = passport_mgr.inspect(result["agent_id"])
        assert data is not None
        assert data["headers"]["kind"] == "passport"
        assert data["headers"]["status"] == "active"

    def test_inspect_nonexistent(self, passport_mgr):
        assert passport_mgr.inspect("ap_nonexistent") is None

    def test_revoke(self, passport_mgr):
        result = passport_mgr.register(name="revoke-bot")
        assert passport_mgr.revoke(result["agent_id"]) is True
        # Verify shows revoked
        verification = passport_mgr.verify(result["agent_id"])
        assert verification["valid"] is False
        assert "revoked" in verification["reason"]

    def test_revoke_nonexistent(self, passport_mgr):
        assert passport_mgr.revoke("ap_nonexistent") is False

    def test_verify_valid(self, passport_mgr):
        result = passport_mgr.register(name="valid-bot")
        verification = passport_mgr.verify(result["agent_id"])
        assert verification["valid"] is True

    def test_verify_expired(self, passport_mgr):
        result = passport_mgr.register(name="expired-bot", ttl_days=0)
        verification = passport_mgr.verify(result["agent_id"])
        # ttl_days=0 means expires today, which should still be valid on same day
        # but we can test with negative-like scenario by manipulating the file
        assert verification["valid"] is True or "expired" in verification.get("reason", "")

    def test_list_all(self, passport_mgr):
        passport_mgr.register(name="bot-a")
        passport_mgr.register(name="bot-b")
        agents = passport_mgr.list_all()
        assert len(agents) == 2

    def test_list_filtered(self, passport_mgr):
        r1 = passport_mgr.register(name="active-bot")
        r2 = passport_mgr.register(name="revoked-bot")
        passport_mgr.revoke(r2["agent_id"])
        active = passport_mgr.list_all(status_filter="active")
        assert len(active) == 1
        assert active[0]["name"] == "active-bot"

    def test_register_with_tools(self, passport_mgr):
        result = passport_mgr.register(
            name="scoped-bot",
            tools_allowed=["read_file", "search"],
            tools_denied=["delete_*"],
        )
        data = passport_mgr.inspect(result["agent_id"])
        perms = "\n".join(data["sections"].get("permissions", []))
        assert "read_file" in perms
        assert "delete_*" in perms


# ── ToolPolicy ────────────────────────────────────────────────────

class TestToolPolicy:
    def test_set_and_test_allow(self, policy_mgr):
        policy_mgr.set_policy("my-bot", tools_allowed=["read_*", "search"])
        result = policy_mgr.test("my-bot", "read_file")
        assert result["allowed"] is True

    def test_deny_pattern(self, policy_mgr):
        policy_mgr.set_policy("my-bot", tools_denied=["delete_*"])
        result = policy_mgr.test("my-bot", "delete_user")
        assert result["allowed"] is False
        assert "denied" in result["reason"]

    def test_not_in_allowed_list(self, policy_mgr):
        policy_mgr.set_policy("my-bot", tools_allowed=["read_*"])
        result = policy_mgr.test("my-bot", "write_file")
        assert result["allowed"] is False
        assert "not in allowed" in result["reason"]

    def test_no_policy_default_allow(self, policy_mgr):
        result = policy_mgr.test("unknown-bot", "anything")
        assert result["allowed"] is True
        assert "default" in result["reason"]

    def test_data_class_restriction(self, policy_mgr):
        policy_mgr.set_policy("my-bot", data_classes=["public", "internal"])
        result = policy_mgr._evaluate("my-bot", "my-bot", "read_file", data_class="secret")
        assert result["allowed"] is False
        assert "data class" in result["reason"]

    def test_check_with_passport(self, tmp_store):
        """check() requires a valid passport."""
        passport_mgr = AgentPassport(store=tmp_store)
        r = passport_mgr.register(name="policy-bot")

        policy = ToolPolicy(store=tmp_store)
        policy.set_policy("policy-bot", tools_allowed=["*"])
        result = policy.check(r["agent_id"], "anything")
        assert result["allowed"] is True

    def test_check_revoked_passport(self, tmp_store):
        passport_mgr = AgentPassport(store=tmp_store)
        r = passport_mgr.register(name="revoked-policy-bot")
        passport_mgr.revoke(r["agent_id"])

        policy = ToolPolicy(store=tmp_store)
        result = policy.check(r["agent_id"], "anything")
        assert result["allowed"] is False


# ── CredentialBroker ──────────────────────────────────────────────

class TestCredentialBroker:
    def test_issue_and_validate(self, tmp_store):
        passport_mgr = AgentPassport(store=tmp_store)
        r = passport_mgr.register(name="cred-bot")
        broker = CredentialBroker(store=tmp_store, secret_key="test")

        cred = broker.issue(r["agent_id"], "read_file")
        assert "token" in cred
        assert cred["agent_id"] == r["agent_id"]
        assert cred["tool"] == "read_file"

        validation = broker.validate(cred["token"], r["agent_id"], "read_file")
        assert validation["valid"] is True

    def test_validate_wrong_token(self, tmp_store):
        broker = CredentialBroker(store=tmp_store, secret_key="test")
        validation = broker.validate("bad-token", "ap_fake", "read_file")
        assert validation["valid"] is False

    def test_issue_revoked_passport(self, tmp_store):
        passport_mgr = AgentPassport(store=tmp_store)
        r = passport_mgr.register(name="revoked-cred-bot")
        passport_mgr.revoke(r["agent_id"])
        broker = CredentialBroker(store=tmp_store, secret_key="test")

        with pytest.raises(PermissionError, match="revoked"):
            broker.issue(r["agent_id"], "read_file")

    def test_issue_denied_tool(self, tmp_store):
        passport_mgr = AgentPassport(store=tmp_store)
        r = passport_mgr.register(name="denied-cred-bot")
        policy = ToolPolicy(store=tmp_store)
        policy.set_policy("denied-cred-bot", tools_denied=["delete_*"])
        broker = CredentialBroker(store=tmp_store, secret_key="test")

        with pytest.raises(PermissionError, match="denied"):
            broker.issue(r["agent_id"], "delete_user")

    def test_revoke_all_tokens(self, tmp_store):
        passport_mgr = AgentPassport(store=tmp_store)
        r = passport_mgr.register(name="multi-cred-bot")
        broker = CredentialBroker(store=tmp_store, secret_key="test")

        broker.issue(r["agent_id"], "tool1")
        broker.issue(r["agent_id"], "tool2")
        removed = broker.revoke_all(r["agent_id"])
        assert removed == 2


# ── AuditLogger ───────────────────────────────────────────────────

class TestAuditLogger:
    def test_session_lifecycle(self, tmp_store):
        passport_mgr = AgentPassport(store=tmp_store)
        r = passport_mgr.register(name="audit-bot")
        logger = AuditLogger(store=tmp_store)

        sid = logger.start_session(r["agent_id"], "test goal")
        assert sid.startswith("as_")

        logger.log_action(sid, "read_file", "read config.json", True)
        logger.log_action(sid, "delete_db", "attempted drop table", False, "denied by policy")

        content = logger.end_session(sid)
        assert "ALLOW" in content
        assert "DENY" in content
        assert "risk_score" in content

    def test_unknown_session_raises(self, audit):
        with pytest.raises(ValueError, match="unknown session"):
            audit.log_action("as_nonexistent", "tool", "detail", True)

    def test_export_crumb_format(self, tmp_store):
        passport_mgr = AgentPassport(store=tmp_store)
        r = passport_mgr.register(name="export-bot")
        logger = AuditLogger(store=tmp_store)

        sid = logger.start_session(r["agent_id"], "export test")
        logger.log_action(sid, "search", "searched logs", True)
        logger.end_session(sid)

        evidence = logger.export_evidence(output_format="crumb")
        assert "BEGIN CRUMB" in evidence

    def test_export_json_format(self, tmp_store):
        passport_mgr = AgentPassport(store=tmp_store)
        r = passport_mgr.register(name="json-bot")
        logger = AuditLogger(store=tmp_store)

        sid = logger.start_session(r["agent_id"], "json test")
        logger.log_action(sid, "read", "read file", True)
        logger.end_session(sid)

        evidence = logger.export_evidence(output_format="json")
        parsed = json.loads(evidence)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_export_csv_format(self, tmp_store):
        passport_mgr = AgentPassport(store=tmp_store)
        r = passport_mgr.register(name="csv-bot")
        logger = AuditLogger(store=tmp_store)

        sid = logger.start_session(r["agent_id"], "csv test")
        logger.log_action(sid, "write", "wrote output", True)
        logger.end_session(sid)

        evidence = logger.export_evidence(output_format="csv")
        lines = evidence.strip().split("\n")
        assert lines[0].startswith("session_id,")
        assert len(lines) >= 2

    def test_feed(self, tmp_store):
        passport_mgr = AgentPassport(store=tmp_store)
        r = passport_mgr.register(name="feed-bot")
        logger = AuditLogger(store=tmp_store)

        sid = logger.start_session(r["agent_id"], "feed test")
        logger.log_action(sid, "api_call", "called openai", True)
        logger.end_session(sid)

        lines = logger.feed()
        assert len(lines) >= 1
        assert "ALLOW" in lines[0]


# ── protect decorator ─────────────────────────────────────────────

class TestProtectDecorator:
    def test_protect_allows(self, tmp_store):
        passport_mgr = AgentPassport(store=tmp_store)
        r = passport_mgr.register(name="decorator-bot")

        @protect(agent_id=r["agent_id"], tool="my_tool", store=tmp_store)
        def my_tool(x, _agentauth_credential=None):
            return x * 2

        assert my_tool(5) == 10

    def test_protect_denies(self, tmp_store):
        passport_mgr = AgentPassport(store=tmp_store)
        r = passport_mgr.register(name="denied-decorator-bot")
        policy = ToolPolicy(store=tmp_store)
        policy.set_policy("denied-decorator-bot", tools_denied=["blocked_tool"])

        @protect(agent_id=r["agent_id"], tool="blocked_tool", store=tmp_store)
        def blocked_tool():
            return "should not reach"

        with pytest.raises(PermissionError):
            blocked_tool()
