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
};
const SUPPORTED_VERSIONS = new Set(["1.1", "1.2"]);
const FOLD_SECTION_RE = /^fold:([^/]+)\/(summary|full)$/;

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
  if (!(headers.kind in REQUIRED_SECTIONS))
    throw new Error(`unknown kind: ${headers.kind}`);

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
        `missing required section for kind=${headers.kind}: ${section} (or fold:${section}/summary + fold:${section}/full)`,
      );
    }
  }

  validateV12Additive(headers, sections);

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
