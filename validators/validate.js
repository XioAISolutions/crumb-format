#!/usr/bin/env node
const fs = require('fs');

const REQUIRED_HEADERS = ['v', 'kind', 'source'];
const REQUIRED_SECTIONS = {
  task: ['goal', 'context', 'constraints'],
  mem: ['consolidated'],
  map: ['project', 'modules'],
};

function parseCrumb(text) {
  const lines = text.split(/\r?\n/).map((line) => line.replace(/\n$/, ''));
  while (lines.length && !lines[0].trim()) lines.shift();
  while (lines.length && !lines[lines.length - 1].trim()) lines.pop();
  if (!lines.length || lines[0] !== 'BEGIN CRUMB') throw new Error('missing BEGIN CRUMB marker');
  if (lines[lines.length - 1] !== 'END CRUMB') throw new Error('missing END CRUMB marker');
  const sepIndex = lines.indexOf('---');
  if (sepIndex === -1) throw new Error('missing header separator ---');

  const headers = {};
  for (const line of lines.slice(1, sepIndex)) {
    if (!line.trim()) continue;
    const idx = line.indexOf('=');
    if (idx === -1) throw new Error(`invalid header line: ${line}`);
    const key = line.slice(0, idx).trim();
    const value = line.slice(idx + 1).trim();
    headers[key] = value;
  }

  for (const key of REQUIRED_HEADERS) {
    if (!(key in headers)) throw new Error(`missing required header: ${key}`);
  }
  if (headers.v !== '1.1') throw new Error('unsupported version');
  if (!(headers.kind in REQUIRED_SECTIONS)) throw new Error(`unknown kind: ${headers.kind}`);

  const sections = {};
  let current = null;
  for (const rawLine of lines.slice(sepIndex + 1, -1)) {
    const line = rawLine.trim();
    if (!line) continue;
    if (line.startsWith('[') && line.endsWith(']')) {
      current = line.slice(1, -1).trim().toLowerCase();
      if (!(current in sections)) sections[current] = [];
      continue;
    }
    if (!current) throw new Error('body content found before first section');
    sections[current].push(line);
  }

  for (const section of REQUIRED_SECTIONS[headers.kind]) {
    if (!(section in sections)) throw new Error(`missing required section for kind=${headers.kind}: ${section}`);
    if (!sections[section].some(Boolean)) throw new Error(`section is empty: ${section}`);
  }

  return { headers, sections };
}

function main() {
  const paths = process.argv.slice(2);
  if (!paths.length) {
    console.error('usage: validate.js <file> [<file> ...]');
    process.exit(2);
  }
  let exitCode = 0;
  for (const path of paths) {
    try {
      const parsed = parseCrumb(fs.readFileSync(path, 'utf8'));
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
