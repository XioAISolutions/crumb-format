# CRUMB REST API

Minimal JSON API for validating, parsing, and rendering CRUMBs.

The server uses only the Python standard library plus the existing CRUMB parser
and renderer from `cli/crumb.py`.

## Quick start

```bash
python api/server.py
```

Custom host and port:

```bash
python api/server.py --host 0.0.0.0 --port 9000
```

## Endpoints

### Health

```bash
curl http://127.0.0.1:8420/health
# {"status":"ok"}
```

### Validate a CRUMB

```bash
curl -X POST http://127.0.0.1:8420/crumb/validate \
  -H "Content-Type: application/json" \
  -d '{"text":"BEGIN CRUMB\nv=1.1\nkind=task\nsource=test\n---\n[goal]\nShip it\n[context]\n- minimal\n[constraints]\n- none\nEND CRUMB"}'
```

### Parse a CRUMB

```bash
curl -X POST http://127.0.0.1:8420/crumb/parse \
  -H "Content-Type: application/json" \
  -d '{"text":"BEGIN CRUMB\nv=1.1\nkind=task\nsource=test\n---\n[goal]\nShip it\n[context]\n- minimal\n[constraints]\n- none\nEND CRUMB"}'
```

### Render a CRUMB

```bash
curl -X POST http://127.0.0.1:8420/crumb/render \
  -H "Content-Type: application/json" \
  -d '{"headers":{"v":"1.1","kind":"task","source":"api-test"},"sections":{"goal":["Ship it"],"context":["- minimal"],"constraints":["- none"]}}'
```

## Notes

- Responses are JSON
- CORS is enabled with `Access-Control-Allow-Origin: *`
- Unknown routes return `404`
- Invalid JSON or missing fields return `400`
- `/crumb/validate` returns `200` with `valid: false` for structurally invalid CRUMBs
