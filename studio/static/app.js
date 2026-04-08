const PREVIEW_HISTORY_KEY = "crumbStudioPreviewHistory";

const state = {
  appVersion: "dev",
  history: [],
  currentResult: null,
  activePreset: null,
  toastTimer: null,
};

const DEMO_PRESETS = {
  "launch-task": {
    mode: "task",
    title: "Resolve launch redirect blocker",
    source: "cursor.thread",
    inputText: [
      "We still have a launch blocker in auth.",
      "The redirect loop happens after refresh when the cookie lands after middleware runs.",
      "Keep the existing cookie names and do not touch the login screen.",
      "Update middleware.ts, stabilize the refresh path, and add a regression check in tests/auth.spec.ts before shipping.",
    ].join("\n"),
  },
  "founder-memory": {
    mode: "mem",
    title: "Founder operating preferences",
    source: "meeting.notes",
    inputText: [
      "Prefers direct technical answers with no fluff.",
      "Always optimize for something that ships this week, not theoretical architecture.",
      "Keep the existing CLI stable.",
      "Avoid platform lock-in and preserve reuse across tools.",
    ].join("\n"),
  },
  "repo-map": {
    mode: "map",
    title: "CRUMB Studio launch surface",
    source: "repo.summary",
    inputText: [
      "Desktop app shell lives in studio/app.py and talks to the existing Python engine.",
      "The transformation layer is in studio/engine.py.",
      "The visual split-pane lives in studio/static/index.html, studio/static/app.css, and studio/static/app.js.",
      "Packaging uses studio/packaging/build_studio.py and should emit release-ready desktop artifacts.",
    ].join("\n"),
  },
  "ops-log": {
    mode: "log",
    title: "Launch readiness sync",
    source: "slack.thread",
    inputText: [
      "09:05 Design polish landed for the split-pane layout.",
      "09:14 Packaged app smoke test failed because the frozen bundle could not import crumb.",
      "09:28 Fixed the bridge to import from cli.crumb and rebuilt the app.",
      "09:41 macOS bundle smoke test passed from dist/CRUMB-Studio.app.",
    ].join("\n"),
  },
  "exec-todo": {
    mode: "todo",
    title: "Next launch actions",
    source: "planning.doc",
    inputText: [
      "Polish the desktop UI for screenshots.",
      "Add release automation for macOS and Windows.",
      "Push the branch and open a PR.",
      "Record a short demo and attach the packaged builds.",
    ].join("\n"),
  },
};

const elements = {
  mode: document.getElementById("modeSelect"),
  title: document.getElementById("titleInput"),
  source: document.getElementById("sourceInput"),
  input: document.getElementById("inputEditor"),
  outputPreview: document.getElementById("outputPreview"),
  outputEmpty: document.getElementById("outputEmptyState"),
  outputSummary: document.getElementById("outputSummary"),
  runButton: document.getElementById("runButton"),
  copyButton: document.getElementById("copyButton"),
  saveButton: document.getElementById("saveButton"),
  exportButton: document.getElementById("exportButton"),
  exportMenu: document.getElementById("exportMenu"),
  clearButton: document.getElementById("clearButton"),
  clearInputButton: document.getElementById("clearInputButton"),
  historyButton: document.getElementById("historyButton"),
  closeHistoryButton: document.getElementById("closeHistoryButton"),
  clearHistoryButton: document.getElementById("clearHistoryButton"),
  historyDrawer: document.getElementById("historyDrawer"),
  historyList: document.getElementById("historyList"),
  modeBadge: document.getElementById("modeBadge"),
  errorBanner: document.getElementById("errorBanner"),
  toast: document.getElementById("toast"),
  statusValue: document.getElementById("statusValue"),
  inputStat: document.getElementById("inputStat"),
  outputStat: document.getElementById("outputStat"),
  ratioStat: document.getElementById("ratioStat"),
  actionNote: document.getElementById("actionNote"),
  sampleChips: Array.from(document.querySelectorAll("[data-preset]")),
};

function bridge() {
  return window.pywebview && window.pywebview.api ? window.pywebview.api : previewBridge;
}

function isPreviewMode() {
  return !(window.pywebview && window.pywebview.api);
}

function slugify(value) {
  return (value || "crumb-studio-output")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "crumb-studio-output";
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function estimateTokens(text) {
  return Math.ceil((text || "").length / 4);
}

function nonEmptyLines(text) {
  return (text || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function summarize(text, limit = 120) {
  const value = (text || "").trim();
  if (value.length <= limit) return value;
  return `${value.slice(0, limit - 1).trim()}…`;
}

function inferConstraints(lines) {
  const matches = lines.filter((line) => /(keep|avoid|preserve|do not|don't|must|cannot|only)/i.test(line));
  if (matches.length) {
    return matches.slice(0, 4).map((line) => `- ${summarize(line, 110)}`);
  }
  return [
    "- Preserve the important behavior already described in the source context.",
    "- Keep the handoff compact enough to paste into another AI tool quickly.",
  ];
}

function buildPreviewCrumb(payload) {
  const mode = payload.mode || "task";
  const title = payload.title || "Preview output";
  const source = payload.source || "crumb.studio.preview";
  const lines = nonEmptyLines(payload.inputText);
  const first = lines[0] || "Source context provided.";

  const crumbLines = [
    "BEGIN CRUMB",
    "v=1.1",
    `kind=${mode}`,
    `title=${title}`,
    `source=${source}`,
  ];

  if (mode === "map") {
    crumbLines.push(`project=${slugify(title)}`);
  }

  crumbLines.push("---");

  if (mode === "mem") {
    crumbLines.push("[consolidated]");
    (lines.slice(0, 6).length ? lines.slice(0, 6) : ["No clear durable memory items were extracted yet."]).forEach((line) => {
      crumbLines.push(`- ${summarize(line, 110)}`);
    });
  } else if (mode === "map") {
    crumbLines.push("[project]");
    crumbLines.push(summarize(first, 150));
    crumbLines.push("");
    crumbLines.push("[modules]");
    (lines.slice(0, 6).length ? lines.slice(0, 6) : ["core workflow", "open questions"]).forEach((line) => {
      crumbLines.push(`- ${summarize(line, 100)}`);
    });
  } else if (mode === "log") {
    crumbLines.push("[entries]");
    const now = new Date().toISOString();
    (lines.slice(0, 6).length ? lines.slice(0, 6) : ["No clear events were extracted."]).forEach((line) => {
      crumbLines.push(`- [${now}] ${summarize(line, 110)}`);
    });
  } else if (mode === "todo") {
    crumbLines.push("[tasks]");
    (lines.slice(0, 6).length ? lines.slice(0, 6) : ["Review the source context and identify the next action."]).forEach((line) => {
      crumbLines.push(`- [ ] ${summarize(line, 110)}`);
    });
  } else {
    crumbLines.push("[goal]");
    crumbLines.push(summarize(first, 140));
    crumbLines.push("");
    crumbLines.push("[context]");
    (lines.slice(0, 6).length ? lines.slice(0, 6) : ["Source context provided, but no strong signals were extracted."]).forEach((line) => {
      crumbLines.push(`- ${summarize(line, 110)}`);
    });
    crumbLines.push("");
    crumbLines.push("[constraints]");
    inferConstraints(lines).forEach((line) => crumbLines.push(line));
  }

  crumbLines.push("END CRUMB");
  return crumbLines.join("\n");
}

function buildPreviewExports(result) {
  const markdown = [
    `# ${result.title}`,
    "",
    `- mode: ${result.mode}`,
    `- source: ${result.source}`,
    "",
    "```text",
    result.outputText,
    "```",
  ].join("\n");

  const json = JSON.stringify(
    {
      mode: result.mode,
      title: result.title,
      source: result.source,
      inputText: result.inputText,
      outputText: result.outputText,
      stats: result.stats,
    },
    null,
    2,
  );

  return { markdown, json };
}

function buildStats(inputText, outputText) {
  const inputChars = inputText.length;
  const outputChars = outputText.length;
  const inputTokens = estimateTokens(inputText);
  const outputTokens = estimateTokens(outputText);

  return {
    input_chars: inputChars,
    output_chars: outputChars,
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    input_lines: nonEmptyLines(inputText).length,
    output_lines: nonEmptyLines(outputText).length,
    char_delta: inputChars - outputChars,
    token_delta: inputTokens - outputTokens,
    output_ratio: Number((outputChars / Math.max(inputChars, 1)).toFixed(3)),
  };
}

function historyItemFromResult(result) {
  return {
    id: result.id,
    createdAt: result.createdAt,
    mode: result.mode,
    title: result.title,
    source: result.source,
    inputPreview: summarize(nonEmptyLines(result.inputText).slice(0, 2).join(" "), 160),
    outputPreview: summarize(nonEmptyLines(result.outputText).slice(0, 3).join(" "), 180),
    inputText: result.inputText,
    outputText: result.outputText,
    stats: result.stats,
  };
}

function readPreviewHistory() {
  try {
    const raw = window.localStorage.getItem(PREVIEW_HISTORY_KEY);
    const payload = raw ? JSON.parse(raw) : [];
    return Array.isArray(payload) ? payload : [];
  } catch {
    return [];
  }
}

function writePreviewHistory(items) {
  try {
    window.localStorage.setItem(PREVIEW_HISTORY_KEY, JSON.stringify(items.slice(0, 20)));
  } catch {
    // localStorage can fail in restricted contexts; ignore.
  }
}

function downloadText(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

const previewBridge = {
  bootstrap: async () => ({
    ok: true,
    appVersion: "preview",
    defaultSource: "crumb.studio.preview",
    modes: ["task", "mem", "map", "log", "todo"],
    history: readPreviewHistory(),
  }),

  generate: async (payload) => {
    const inputText = String(payload.inputText || "").trim();
    if (!inputText) {
      return { ok: false, error: "Paste some raw context before generating." };
    }

    const result = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2, 10)}`,
      createdAt: new Date().toISOString(),
      mode: String(payload.mode || "task"),
      title: String(payload.title || "").trim() || "Preview output",
      source: String(payload.source || "").trim() || "crumb.studio.preview",
      inputText,
    };

    result.outputText = buildPreviewCrumb(result);
    result.stats = buildStats(result.inputText, result.outputText);
    const exports = buildPreviewExports(result);
    result.outputMarkdown = exports.markdown;
    result.outputJson = exports.json;

    const history = [historyItemFromResult(result), ...readPreviewHistory().filter((item) => item.id !== result.id)].slice(0, 20);
    writePreviewHistory(history);
    return { ok: true, result, history };
  },

  copy_output: async (outputText) => {
    try {
      await navigator.clipboard.writeText(outputText);
      return { ok: true, message: "Copied to clipboard." };
    } catch {
      return { ok: false, error: "Clipboard access is unavailable in this browser." };
    }
  },

  save_output: async (payload) => {
    const title = String(payload.title || "").trim() || "crumb-studio-output";
    const outputText = String(payload.outputText || "");
    const filename = `${slugify(title)}.crumb`;
    downloadText(filename, outputText, "text/plain;charset=utf-8");
    return { ok: true, path: filename };
  },

  export_output: async (payload) => {
    const title = String(payload.title || "").trim() || "crumb-studio-output";
    const outputText = String(payload.outputText || "");
    const format = String(payload.format || "plain");

    let filename = `${slugify(title)}.txt`;
    let content = outputText;
    let mimeType = "text/plain;charset=utf-8";

    if (format === "markdown") {
      filename = `${slugify(title)}.md`;
      content = `# ${title}\n\n\`\`\`text\n${outputText}\n\`\`\`\n`;
      mimeType = "text/markdown;charset=utf-8";
    } else if (format === "json") {
      filename = `${slugify(title)}.json`;
      content = JSON.stringify({ title, outputText }, null, 2);
      mimeType = "application/json;charset=utf-8";
    }

    downloadText(filename, content, mimeType);
    return { ok: true, path: filename, format };
  },

  get_history: async () => ({ ok: true, history: readPreviewHistory() }),

  load_history_item: async (itemId) => {
    const item = readPreviewHistory().find((entry) => entry.id === itemId);
    return item ? { ok: true, item } : { ok: false, error: "That history item is no longer available." };
  },

  clear_history: async () => {
    writePreviewHistory([]);
    return { ok: true, history: [] };
  },
};

function showToast(message, isError = false) {
  if (!message) return;
  window.clearTimeout(state.toastTimer);
  elements.toast.textContent = message;
  elements.toast.style.borderColor = isError ? "rgba(255, 141, 149, 0.34)" : "rgba(255, 255, 255, 0.14)";
  elements.toast.classList.add("show");
  state.toastTimer = window.setTimeout(() => {
    elements.toast.classList.remove("show");
  }, 2200);
}

function setBusy(isBusy) {
  elements.runButton.disabled = isBusy;
  elements.runButton.textContent = isBusy ? "Generating…" : "Generate";
}

function setActivePreset(presetKey) {
  state.activePreset = presetKey || null;
  elements.sampleChips.forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.preset === state.activePreset);
  });
}

function renderStats(stats) {
  if (!stats) {
    elements.inputStat.textContent = `${elements.input.value.length} chars`;
    elements.outputStat.textContent = "0 chars";
    elements.ratioStat.textContent = "-";
    return;
  }

  const direction = stats.char_delta >= 0 ? "tighter" : "expanded";
  const ratio = stats.output_ratio ? `${(stats.output_ratio * 100).toFixed(0)}%` : "-";
  elements.inputStat.textContent = `${stats.input_chars} chars`;
  elements.outputStat.textContent = `${stats.output_chars} chars`;
  elements.ratioStat.textContent = `${ratio} · ${direction}`;
}

function updateInputStatOnly() {
  if (!state.currentResult) {
    elements.inputStat.textContent = `${elements.input.value.length} chars`;
  }
}

function applyResult(result) {
  state.currentResult = result;
  elements.mode.value = result.mode;
  elements.modeBadge.textContent = result.mode;
  elements.title.value = result.title || "";
  elements.source.value = result.source || "";
  elements.outputPreview.textContent = result.outputText;
  elements.outputPreview.classList.remove("hidden");
  elements.outputEmpty.classList.add("hidden");
  elements.copyButton.disabled = false;
  elements.saveButton.disabled = false;
  elements.exportButton.disabled = false;
  elements.statusValue.textContent = "Generated";
  elements.outputSummary.textContent = `${result.stats.output_lines} lines · ${result.stats.output_tokens} tok`;
  elements.actionNote.textContent = "Ready to copy, save, or export.";
  renderStats(result.stats);
}

function clearError() {
  elements.errorBanner.textContent = "";
  elements.errorBanner.classList.add("hidden");
}

function showError(message) {
  elements.errorBanner.textContent = message;
  elements.errorBanner.classList.remove("hidden");
  elements.statusValue.textContent = "Error";
}

function clearResult() {
  state.currentResult = null;
  elements.outputPreview.textContent = "";
  elements.outputPreview.classList.add("hidden");
  elements.outputEmpty.classList.remove("hidden");
  elements.copyButton.disabled = true;
  elements.saveButton.disabled = true;
  elements.exportButton.disabled = true;
  elements.statusValue.textContent = isPreviewMode() ? "Preview" : "Ready";
  elements.outputSummary.textContent = "No output yet";
  elements.actionNote.textContent = isPreviewMode()
    ? "Preview mode supports copy, save, export, and local history."
    : "Cmd/Ctrl + Enter runs the transformation.";
  renderStats(null);
}

async function loadPreset(presetKey, autoRun = false) {
  const preset = DEMO_PRESETS[presetKey];
  if (!preset) return;

  setActivePreset(presetKey);
  elements.mode.value = preset.mode;
  elements.title.value = preset.title;
  elements.source.value = preset.source;
  elements.input.value = preset.inputText;
  clearError();
  clearResult();
  updateInputStatOnly();

  if (autoRun) {
    await runGeneration();
  }
}

function renderHistory(items) {
  state.history = items || [];
  elements.historyList.innerHTML = "";

  if (!state.history.length) {
    elements.historyList.innerHTML = `
      <div class="empty-state">
        <strong>No history yet.</strong>
        <p>Generated outputs will appear here for quick reload.</p>
      </div>
    `;
    return;
  }

  state.history.forEach((item) => {
    const button = document.createElement("button");
    button.className = "history-item";
    button.innerHTML = `
      <strong>${escapeHtml(item.title || "Untitled output")}</strong>
      <small>${escapeHtml(item.mode || "task")} · ${escapeHtml(item.createdAt || "")}</small>
      <p>${escapeHtml(item.inputPreview || "")}</p>
    `;
    button.addEventListener("click", async () => {
      const response = await bridge().load_history_item(item.id);
      if (!response.ok) {
        showToast(response.error || "Unable to load that history item.", true);
        return;
      }

      const entry = response.item;
      setActivePreset(null);
      elements.input.value = entry.inputText || "";
      elements.title.value = entry.title || "";
      elements.source.value = entry.source || "crumb.studio";
      elements.mode.value = entry.mode || "task";
      applyResult({
        id: entry.id,
        createdAt: entry.createdAt,
        mode: entry.mode,
        title: entry.title,
        source: entry.source,
        inputText: entry.inputText,
        outputText: entry.outputText,
        stats: entry.stats || buildStats(entry.inputText || "", entry.outputText || ""),
      });
      toggleHistory(false);
    });
    elements.historyList.appendChild(button);
  });
}

function toggleHistory(show) {
  elements.historyDrawer.classList.toggle("hidden", !show);
}

async function runGeneration() {
  clearError();
  setBusy(true);
  const response = await bridge().generate({
    inputText: elements.input.value,
    mode: elements.mode.value,
    title: elements.title.value,
    source: elements.source.value,
  });
  setBusy(false);

  if (!response.ok) {
    clearResult();
    showError(response.error || "Generation failed.");
    return;
  }

  applyResult(response.result);
  renderHistory(response.history || []);
  showToast("Structured CRUMB generated.");
}

async function copyOutput() {
  if (!state.currentResult) return;
  const response = await bridge().copy_output(state.currentResult.outputText);
  if (!response.ok) {
    showToast(response.error || "Copy failed.", true);
    return;
  }
  showToast(response.message || "Copied.");
}

async function saveOutput() {
  if (!state.currentResult) return;
  const response = await bridge().save_output({
    title: state.currentResult.title,
    outputText: state.currentResult.outputText,
  });
  if (response.cancelled) return;
  if (!response.ok) {
    showToast(response.error || "Save failed.", true);
    return;
  }
  showToast(`Saved ${response.path}`);
}

async function exportOutput(format) {
  if (!state.currentResult) return;
  const response = await bridge().export_output({
    title: state.currentResult.title,
    outputText: state.currentResult.outputText,
    format,
  });
  if (response.cancelled) return;
  if (!response.ok) {
    showToast(response.error || "Export failed.", true);
    return;
  }
  showToast(`Exported ${response.format} to ${response.path}`);
}

async function clearHistory() {
  const response = await bridge().clear_history();
  if (!response.ok) {
    showToast(response.error || "Unable to clear history.", true);
    return;
  }
  renderHistory(response.history || []);
  showToast("History cleared.");
}

function resetAll() {
  setActivePreset(null);
  elements.input.value = "";
  elements.title.value = "";
  clearError();
  clearResult();
  updateInputStatOnly();
}

function wireEvents() {
  elements.runButton.addEventListener("click", runGeneration);
  elements.copyButton.addEventListener("click", copyOutput);
  elements.saveButton.addEventListener("click", saveOutput);
  elements.clearButton.addEventListener("click", resetAll);
  elements.clearInputButton.addEventListener("click", () => {
    elements.input.value = "";
    setActivePreset(null);
    clearError();
    updateInputStatOnly();
  });
  elements.historyButton.addEventListener("click", () => toggleHistory(true));
  elements.closeHistoryButton.addEventListener("click", () => toggleHistory(false));
  elements.clearHistoryButton.addEventListener("click", clearHistory);
  elements.sampleChips.forEach((chip) => {
    chip.addEventListener("click", async () => {
      await loadPreset(chip.dataset.preset, true);
    });
  });
  elements.exportButton.addEventListener("click", () => {
    elements.exportMenu.classList.toggle("hidden");
  });

  elements.exportMenu.querySelectorAll("[data-export-format]").forEach((button) => {
    button.addEventListener("click", async () => {
      elements.exportMenu.classList.add("hidden");
      await exportOutput(button.dataset.exportFormat);
    });
  });

  elements.input.addEventListener("input", () => {
    setActivePreset(null);
    updateInputStatOnly();
  });

  elements.mode.addEventListener("change", () => {
    setActivePreset(null);
  });

  document.addEventListener("click", (event) => {
    if (!elements.exportMenu.contains(event.target) && event.target !== elements.exportButton) {
      elements.exportMenu.classList.add("hidden");
    }
  });

  document.addEventListener("keydown", async (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      await runGeneration();
    }
    if (event.key === "Escape") {
      toggleHistory(false);
      elements.exportMenu.classList.add("hidden");
    }
  });
}

async function bootstrap() {
  const response = await bridge().bootstrap();
  if (!response.ok) return;

  state.appVersion = response.appVersion || "dev";
  elements.source.value = response.defaultSource || "crumb.studio";
  renderHistory(response.history || []);
  clearResult();
  updateInputStatOnly();

  if (isPreviewMode()) {
    elements.statusValue.textContent = "Preview";
    elements.actionNote.textContent = "Preview mode supports copy, save, export, and local history.";
  }
}

wireEvents();

if (window.pywebview) {
  window.addEventListener("pywebviewready", bootstrap);
} else {
  bootstrap();
}
