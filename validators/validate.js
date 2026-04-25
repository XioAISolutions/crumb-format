#!/usr/bin/env node
const fs = require("fs");

const REQUIRED_HEADERS = ["v", "kind", "source"];
const REQUIRED_SECTIONS = {
  task: ["goal", "context", "constraints"],
  mem: ["consolidated"],
  map: ["project", "modules"],
  log: ["entries"],
  todo: ["tasks"],
  wake: ["identity"],
  delta: ["changes"],
  agent: ["identity"],
};
const SUPPORTED_VERSIONS = new Set(["1.1", "1.2", "1.3"]);
const FOLD_SECTION_RE = /^fold:([^/]+)\/(summary|full)$/;
const HANDOFF_ID_RE = /^[a-zA-Z0-9_-]+$/;
const WORKFLOW_LINE_RE = /^\s*-?\s*(\d+)[.)]\s*(.+)$/;
const KV_RE = /([a-zA-Z_][a-zA-Z0-9_]*)=([^\s]+)/g;

function parseCrumb(text) {
  const lines = text.split(/\r?\n/).map((line) => line.replace(/\n$/, ""));
  while (lines.length && !lines[0].trim()) lines.shift();
  while (lines.length && !lines[lines.length - 1].trim()) lines.pop();
  if (!lines.length || lines[0] !== "BEGIN CRUMB")
    throw new Error("missing BEGIN CRUMB marker");
  if (lines[lines.length - 1] !== "END CRUMB")
    throw new Error("missing END CRUMB marker");
  const sepIndex = lines.indexOf("---");
  if (sepIndex === -1) throw new Error("missing header separator ---");

  const headers = {};
  for (const line of lines.slice(1, sepIndex)) {
    if (!line.trim()) continue;
    const idx = line.indexOf("=");
    if (idx === -1) throw new Error(`invalid header line: ${line}`);
    const key = line.slice(0, idx).trim();
    const value = line.slice(idx + 1).trim();
    headers[key] = value;
  }

  for (const key of REQUIRED_HEADERS) {
    if (!(key in headers)) throw new Error(`missing required header: ${key}`);
  }
  if (!SUPPORTED_VERSIONS.has(headers.v))
    throw new Error(`unsupported version: ${headers.v}`);
  if (!(headers.kind in REQUIRED_SECTIONS)) {
    const valid = Object.keys(REQUIRED_SECTIONS).sort().join(", ");
    throw new Error(`unknown kind: '${headers.kind}'. valid: ${valid}`);
  }

  const sections = {};
  let current = null;
  for (const rawLine of lines.slice(sepIndex + 1, -1)) {
    const line = rawLine.trim();
    if (!line) continue;
    if (line.startsWith("[") && line.endsWith("]")) {
      current = line.slice(1, -1).trim().toLowerCase();
      if (!(current in sections)) sections[current] = [];
      continue;
    }
    if (!current) throw new Error("body content found before first section");
    sections[current].push(line);
  }

  for (const section of REQUIRED_SECTIONS[headers.kind]) {
    const foldSummary = `fold:${section}/summary`;
    const foldFull = `fold:${section}/full`;
    if (section in sections) {
      if (!sections[section].some(Boolean))
        throw new Error(`section is empty: ${section}`);
    } else if (foldSummary in sections || foldFull in sections) {
      for (const variant of [foldSummary, foldFull]) {
        if (variant in sections && !sections[variant].some(Boolean))
          throw new Error(`section is empty: ${variant}`);
      }
    } else {
      throw new Error(
        `missing required section for kind=${headers.kind}: ${section}`,
      );
    }
  }

  validateV12Additive(headers, sections);
  validateV13Additive(headers, sections);

  return { headers, sections };
}

function validateV12Additive(headers, sections) {
  if ("refs" in headers) {
    const refsValue = headers.refs.trim();
    if (!refsValue)
      throw new Error("refs header must not be empty when present");
    for (const ref of refsValue.split(",").map((r) => r.trim())) {
      if (!ref) throw new Error("refs header contains an empty entry");
    }
  }
  if ("refs" in sections && !sections.refs.some((line) => line.trim()))
    throw new Error("[refs] section is empty; omit it instead");
  if ("handoff" in sections && !sections.handoff.some((line) => line.trim()))
    throw new Error("[handoff] section is empty; omit it instead");

  const foldPairs = {};
  for (const name of Object.keys(sections)) {
    const matched = name.match(FOLD_SECTION_RE);
    if (!matched) continue;
    const [, foldName, variant] = matched;
    if (!foldPairs[foldName]) foldPairs[foldName] = new Set();
    foldPairs[foldName].add(variant);
  }
  for (const [foldName, variants] of Object.entries(foldPairs)) {
    if (variants.has("full") && !variants.has("summary"))
      throw new Error(
        `fold:${foldName} declares /full without a paired /summary`,
      );
  }

  for (const [name, body] of Object.entries(sections)) {
    const first = body.find((line) => line.trim()) || "";
    if (first.trim().startsWith("@type:")) {
      const value = first.trim().slice("@type:".length).trim();
      if (!value)
        throw new Error(`@type annotation has empty value in [${name}]`);
    }
  }
}

function parseKvLine(line) {
  let body = line.trim();
  if (body.startsWith("- ")) body = body.slice(2);
  else if (body.startsWith("-")) body = body.slice(1);
  const tokens = {};
  for (const m of body.matchAll(KV_RE)) {
    tokens[m[1]] = m[2];
  }
  return tokens;
}

function detectDepCycle(deps, label) {
  const color = {};
  for (const k of Object.keys(deps)) color[k] = 0;
  function visit(node) {
    const c = color[node] ?? 0;
    if (c === 1) throw new Error(`${label} dependency cycle through '${node}'`);
    if (c === 2) return;
    color[node] = 1;
    for (const child of deps[node] || []) {
      if (child in deps) visit(child);
    }
    color[node] = 2;
  }
  for (const node of Object.keys(deps)) {
    if ((color[node] ?? 0) === 0) visit(node);
  }
}

function validateV13Additive(headers, sections) {
  if ("fold_priority" in headers) {
    const value = headers.fold_priority.trim();
    if (!value) throw new Error("fold_priority header must not be empty when present");
    for (const name of value.split(",").map((n) => n.trim())) {
      if (!name) throw new Error("fold_priority contains an empty entry");
      if (!/^[a-zA-Z0-9_-]+$/.test(name))
        throw new Error(`fold_priority entry '${name}' has invalid characters`);
    }
  }
  if ("handoff" in sections) {
    const stepIds = {};
    const deps = {};
    let position = 0;
    for (const line of sections.handoff) {
      const stripped = line.trim();
      if (!stripped || stripped.startsWith("- [x]")) continue;
      if (!stripped.startsWith("-")) continue;
      position += 1;
      const tokens = parseKvLine(stripped);
      const stepId = tokens.id ?? String(position);
      if (!HANDOFF_ID_RE.test(stepId))
        throw new Error(`[handoff] id='${stepId}' must match [a-zA-Z0-9_-]+`);
      if (stepId in stepIds) throw new Error(`[handoff] duplicate id='${stepId}'`);
      stepIds[stepId] = position;
      const after = tokens.after;
      if (after) {
        deps[stepId] = after.split(",").map((d) => d.trim()).filter(Boolean);
      }
    }
    for (const [stepId, refs] of Object.entries(deps)) {
      for (const ref of refs) {
        if (!(ref in stepIds))
          throw new Error(
            `[handoff] id='${stepId}' has unknown after= dependency '${ref}'`,
          );
      }
    }
    detectDepCycle(deps, "[handoff]");
  }
  if ("workflow" in sections) {
    const stepIds = {};
    const deps = {};
    for (const line of sections.workflow) {
      const stripped = line.trim();
      if (!stripped) continue;
      const match = stripped.match(WORKFLOW_LINE_RE);
      if (!match) {
        if (stripped.startsWith("-")) continue;
        throw new Error(`[workflow] line must be numbered: ${stripped}`);
      }
      const num = match[1];
      const rest = match[2];
      const tokens = parseKvLine("- " + rest);
      const stepId = tokens.id ?? num;
      if (!HANDOFF_ID_RE.test(stepId))
        throw new Error(`[workflow] id='${stepId}' must match [a-zA-Z0-9_-]+`);
      if (stepId in stepIds) throw new Error(`[workflow] duplicate id='${stepId}'`);
      stepIds[stepId] = parseInt(num, 10);
      const depends = tokens.depends_on;
      if (depends) {
        deps[stepId] = depends.split(",").map((d) => d.trim()).filter(Boolean);
      }
    }
    for (const [stepId, refs] of Object.entries(deps)) {
      for (const ref of refs) {
        if (!(ref in stepIds))
          throw new Error(
            `[workflow] id='${stepId}' has unknown depends_on '${ref}'`,
          );
      }
    }
    detectDepCycle(deps, "[workflow]");
  }
  if ("script" in sections) {
    const meaningful = sections.script.filter((line) => line.trim());
    if (meaningful.length && !meaningful[0].trim().startsWith("@type:"))
      throw new Error("[script] section must begin with @type: <lang>");
  }
  if ("checks" in sections) {
    for (const line of sections.checks) {
      const stripped = line.trim();
      if (!stripped || !stripped.startsWith("-")) continue;
      const body = stripped.slice(1).trim();
      if (!body.includes("::"))
        throw new Error(
          `[checks] line must use 'name :: status' format: ${stripped}`,
        );
    }
  }
}

function expandArgs(args) {
  const paths = [];
  for (const arg of args) {
    if (fs.existsSync(arg) && fs.statSync(arg).isDirectory()) {
      const stack = [arg];
      while (stack.length) {
        const current = stack.pop();
        for (const entry of fs.readdirSync(current)) {
          const full = `${current}/${entry}`;
          const stat = fs.statSync(full);
          if (stat.isDirectory()) {
            stack.push(full);
          } else if (full.endsWith(".crumb")) {
            paths.push(full);
          }
        }
      }
    } else {
      paths.push(arg);
    }
  }
  return paths;
}

function main() {
  const paths = expandArgs(process.argv.slice(2));
  if (!paths.length) {
    console.error("usage: validate.js <file|dir> [...]");
    process.exit(2);
  }
  let exitCode = 0;
  for (const path of paths) {
    try {
      const parsed = parseCrumb(fs.readFileSync(path, "utf8"));
      console.log(`OK  ${path}  kind=${parsed.headers.kind}`);
    } catch (error) {
      console.error(`ERR ${path}  ${error.message}`);
      exitCode = 1;
    }
  }
  process.exit(exitCode);
}

if (require.main === module) {
  main();
}
