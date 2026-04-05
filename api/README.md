# Crumb API Server

Lightweight REST API for crumb-format and agentauth. Built with **zero external dependencies** -- uses only the Python standard library (`http.server`, `json`).

## Quick Start

```bash
# Default port (8420)
python api/server.py

# Custom port
python api/server.py --port 9000
```

## Endpoints

### Health

```bash
curl http://localhost:8420/health
# {"status": "ok", "version": "0.2.0"}
```

---

### CRUMB Operations

**Validate a crumb document:**

```bash
curl -X POST http://localhost:8420/crumb/validate \
  -H "Content-Type: application/json" \
  -d '{"text": "BEGIN CRUMB\nv=1.1\nkind=task\nsource=test\n---\n[goal]\n  Build something\n[context]\n  Some context\n[constraints]\n  None\nEND CRUMB"}'
# {"valid": true, "error": null}
```

**Parse a crumb document to JSON:**

```bash
curl -X POST http://localhost:8420/crumb/parse \
  -H "Content-Type: application/json" \
  -d '{"text": "BEGIN CRUMB\nv=1.1\nkind=task\nsource=test\n---\n[goal]\n  Build something\n[context]\n  Some context\n[constraints]\n  None\nEND CRUMB"}'
# {"headers": {"v": "1.1", "kind": "task", ...}, "sections": {"goal": [...], ...}}
```

**Render JSON back to crumb text:**

```bash
curl -X POST http://localhost:8420/crumb/render \
  -H "Content-Type: application/json" \
  -d '{
    "headers": {"v": "1.1", "kind": "task", "source": "api-test"},
    "sections": {"goal": ["  Build a REST API"], "context": ["  For crumb-format"], "constraints": ["  No external deps"]}
  }'
# {"text": "BEGIN CRUMB\nv=1.1\n...END CRUMB\n"}
```

---

### Passport Management

**Register a new agent passport:**

```bash
curl -X POST http://localhost:8420/passport/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-agent",
    "framework": "langchain",
    "owner": "team-alpha",
    "tools_allowed": ["read_file", "write_file"],
    "tools_denied": ["exec_shell"],
    "ttl_days": 30
  }'
# {"agent_id": "ap_abc12345", "name": "my-agent", "passport_path": "..."}
```

**Inspect a passport:**

```bash
curl http://localhost:8420/passport/ap_abc12345
# {"headers": {...}, "sections": {...}}
```

**Verify passport validity:**

```bash
curl http://localhost:8420/passport/ap_abc12345/verify
# {"valid": true, "reason": "valid", "passport": {...}}
```

**Revoke a passport:**

```bash
curl -X POST http://localhost:8420/passport/ap_abc12345/revoke
# {"revoked": true, "agent_id": "ap_abc12345"}
```

**List all passports:**

```bash
# All passports
curl http://localhost:8420/passports

# Filter by status
curl "http://localhost:8420/passports?status=active"
curl "http://localhost:8420/passports?status=revoked"
# {"passports": [...], "count": 3}
```

---

### Policy Management

**Set a tool policy for an agent:**

```bash
curl -X POST http://localhost:8420/policy/set \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "my-agent",
    "tools_allowed": ["read_file", "write_file", "search"],
    "tools_denied": ["exec_shell", "sudo_*"],
    "data_classes": ["public", "internal"],
    "max_actions_per_session": 500
  }'
# {"agent_name": "my-agent", "tools_allowed": [...], ...}
```

**Check if a tool is allowed for an agent:**

```bash
curl -X POST http://localhost:8420/policy/check \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "ap_abc12345", "tool": "read_file"}'
# {"allowed": true, "reason": "policy allows", "tool": "read_file", "agent_id": "ap_abc12345"}
```

---

### Credential Broker

**Issue a short-lived credential:**

```bash
curl -X POST http://localhost:8420/credential/issue \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "ap_abc12345", "tool": "read_file", "ttl_seconds": 600}'
# {"token": "...", "agent_id": "ap_abc12345", "tool": "read_file", "expires": "..."}
```

**Validate a credential:**

```bash
curl -X POST http://localhost:8420/credential/validate \
  -H "Content-Type: application/json" \
  -d '{"token": "abc123...", "agent_id": "ap_abc12345", "tool": "read_file"}'
# {"valid": true, "agent_id": "ap_abc12345", "tool": "read_file", "expires": "..."}
```

---

### Audit Trail

**Start an audit session:**

```bash
curl -X POST http://localhost:8420/audit/start \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "ap_abc12345", "goal": "Summarize quarterly report"}'
# {"session_id": "as_def67890", "agent_id": "ap_abc12345"}
```

**Log an action:**

```bash
curl -X POST http://localhost:8420/audit/log \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "as_def67890",
    "tool": "read_file",
    "detail": "Read quarterly_report.pdf",
    "allowed": true,
    "reason": "policy allows"
  }'
# {"logged": true, "session_id": "as_def67890"}
```

**End an audit session:**

```bash
curl -X POST http://localhost:8420/audit/end \
  -H "Content-Type: application/json" \
  -d '{"session_id": "as_def67890", "status": "completed"}'
# {"session_id": "as_def67890", "status": "completed", "crumb": "BEGIN CRUMB\n..."}
```

**Export audit evidence:**

```bash
# As crumb format (default)
curl "http://localhost:8420/audit/export?agent_id=ap_abc12345&format=crumb"

# As JSON
curl "http://localhost:8420/audit/export?agent_id=ap_abc12345&format=json"

# As CSV, filtered by date
curl "http://localhost:8420/audit/export?agent_id=ap_abc12345&since=2025-01-01&format=csv"
# {"format": "json", "data": [...]}
```

**Audit feed (recent actions):**

```bash
curl "http://localhost:8420/audit/feed?agent_id=ap_abc12345"
# {"entries": ["[ap_abc12345/as_def67890] [2025-...] ALLOW read_file: ..."], "count": 5}
```

---

## Notes

- All responses include CORS headers (`Access-Control-Allow-Origin: *`)
- All request/response bodies are JSON (`Content-Type: application/json`)
- Error responses use the format `{"error": "message"}`
- HTTP status codes: 200 (OK), 201 (Created), 400 (Bad Request), 403 (Forbidden), 404 (Not Found), 500 (Internal Server Error)
- Data is stored in `.crumb-auth/` in the working directory (file-based, no database needed)
