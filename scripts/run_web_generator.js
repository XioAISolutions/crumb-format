#!/usr/bin/env node
// Test harness: loads web/index.html, executes its inline <script> inside a
// minimal DOM-shim sandbox, and prints the three kinds' output to stdout in
// the form:
//
//   ===KIND=task===
//   <crumb body>
//   ===KIND=mem===
//   <crumb body>
//   ===KIND=map===
//   <crumb body>
//
// Reads input chat text from stdin. Exit 0 on success, nonzero on error.

const fs = require("fs");
const path = require("path");
const vm = require("vm");

function readStdin() {
  return new Promise(function (resolve, reject) {
    const chunks = [];
    process.stdin.on("data", function (c) { chunks.push(c); });
    process.stdin.on("end", function () { resolve(Buffer.concat(chunks).toString("utf8")); });
    process.stdin.on("error", reject);
  });
}

function makeStub() {
  const stub = {
    value: "",
    textContent: "",
    addEventListener: function () {},
    select: function () {},
    click: function () {},
    appendChild: function () {},
    removeChild: function () {},
    classList: { add: function () {}, remove: function () {} },
    style: {},
    setAttribute: function () {},
  };
  stub.focus = function () {};
  return stub;
}

async function main() {
  const input = await readStdin();
  const htmlPath = path.join(__dirname, "..", "web", "index.html");
  const html = fs.readFileSync(htmlPath, "utf8");
  const m = html.match(/<script>([\s\S]+?)<\/script>/);
  if (!m) {
    console.error("no <script> in web/index.html");
    process.exit(2);
  }

  const docStub = {
    getElementById: function () { return makeStub(); },
    createElement: function () { return makeStub(); },
    body: { appendChild: function () {}, removeChild: function () {} },
    execCommand: function () {},
  };
  const sandbox = {
    window: {},
    document: docStub,
    navigator: { clipboard: { writeText: function () { return Promise.resolve(); } } },
    setTimeout: setTimeout,
    clearTimeout: clearTimeout,
    Blob: function () {},
    URL: { createObjectURL: function () { return ""; }, revokeObjectURL: function () {} },
    console: console,
  };
  vm.createContext(sandbox);
  vm.runInContext(m[1], sandbox);

  const gen = sandbox.window.__crumbNew;
  if (!gen || !gen.generateTask) {
    console.error("window.__crumbNew was not exported from web/index.html");
    process.exit(3);
  }

  process.stdout.write("===KIND=task===\n" + gen.generateTask(input) + "\n");
  process.stdout.write("===KIND=mem===\n"  + gen.generateMem(input)  + "\n");
  process.stdout.write("===KIND=map===\n"  + gen.generateMap(input)  + "\n");
}

main().catch(function (err) {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
