const state = {
  appVersion: "dev",
  history: [],
  currentResult: null,
  toastTimer: null,
};

const DEMO_PRESETS = {
  "launch-task": {
    mode: "task",
    title: "Resolve launch redirect blocker",
    source: "cursor.thread",
    inputText: [
      "user: We still have a launch blocker in auth.",
      "assistant: The redirect loop happens after refresh when the cookie lands after middleware runs.",
      "user: Keep the existing cookie names and do not touch the login screen.",
      "assistant: Update middleware.ts, stabilize the refresh path, and add a regression check in tests/auth.spec.ts before shipping.",
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
      "Important: screenshots and demos need to look premium.",
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
      "09:52 Next step is Windows packaging automation.",
    ].join("\n"),
  },
  "exec-todo": {
    mode: "todo",
    title: "Next launch actions",
    source: "planning.doc",
    inputText: [
      "Need to polish the desktop UI for screenshots.",
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
  resultStrip: document.getElementById("resultStrip"),
  resultFlow: document.getElementById("resultFlow"),
  resultShape: document.getElementById("resultShape"),
  resultTargets: document.getElementById("resultTargets"),
  toast: document.getElementById("toast"),
  statusValue: document.getElementById("statusValue"),
  inputStat: document.getElementById("inputStat"),
  outputStat: document.getElementById("outputStat"),
  ratioStat: document.getElementById("ratioStat"),
  actionNote: document.getElementById("actionNote"),
  presetCards: Array.from(document.querySelectorAll("[data-preset]")),
};

const previewBridge = {
  bootstrap: async () => ({
    ok: true,
    appVersion: "preview",
    defaultSource: "crumb.studio.preview",
    modes: ["task", "mem", "map", "log", "todo"],
    history: [],
  }),
  generate: async (payload) => {
    if (!payload.inputText.trim()) {
      return { ok: false, error: "Paste some raw context before running the preview." };
    }
    const title = payload.title || "Preview output";
    const outputText = [
      "BEGIN CRUMB",
      "v=1.1",
      `kind=${payload.mode}`,
      `title=${title}`,
      `source=${payload.source || "crumb.studio.preview"}`,
      "---",
      payload.mode === "mem" ? "[consolidated]" : "[goal]",
      payload.mode === "mem" ? "- Preview mode can show the UI without the Python bridge." : "Preview mode only — run inside CRUMB Studio for the real Python engine.",
      "",
      payload.mode === "mem" ? "" : "[context]",
      payload.mode === "mem" ? "" : `- ${payload.inputText.trim().split("\n")[0].slice(0, 120)}`,
      payload.mode === "mem" ? "" : "",
      payload.mode === "mem" ? "" : "[constraints]",
      payload.mode === "mem" ? "" : "- Launch the desktop runtime to use the real engine.",
      "END CRUMB",
    ].filter(Boolean).join("\n");
    return {
      ok: true,
      result: {
        id: "preview",
        createdAt: new Date().toISOString(),
        mode: payload.mode,
        title,
        source: payload.source || "crumb.studio.preview",
        inputText: payload.inputText,
        outputText,
        outputMarkdown: outputText,
        outputJson: JSON.stringify({ preview: true }, null, 2),
        stats: {
          input_chars: payload.inputText.length,
          output_chars: outputText.length,
          input_tokens: Math.floor(payload.inputText.length / 4),
          output_tokens: Math.floor(outputText.length / 4),
          input_lines: payload.inputText.split("\n").filter(Boolean).length,
          output_lines: outputText.split("\n").filter(Boolean).length,
          char_delta: payload.inputText.length - outputText.length,
          token_delta: Math.floor(payload.inputText.length / 4) - Math.floor(outputText.length / 4),
          output_ratio: Number((outputText.length / Math.max(payload.inputText.length, 1)).toFixed(3)),
        },
      },
      history: [],
    };
  },
  copy_output: async () => ({ ok: false, error: "Clipboard is disabled in preview mode." }),
  save_output: async () => ({ ok: false, error: "Desktop save is disabled in preview mode." }),
  export_output: async () => ({ ok: false, error: "Desktop export is disabled in preview mode." }),
  get_history: async () => ({ ok: true, history: [] }),
  load_history_item: async () => ({ ok: false, error: "Preview mode has no saved history." }),
  clear_history: async () => ({ ok: true, history: [] }),
};

const bridge = () => (window.pywebview && window.pywebview.api ? window.pywebview.api : previewBridge);

function showToast(message, isError = false) {
  if (!message) return;
  window.clearTimeout(state.toastTimer);
  elements.toast.textContent = message;
  elements.toast.style.borderColor = isError ? "rgba(255, 127, 134, 0.34)" : "";
  elements.toast.classList.add("show");
  state.toastTimer = window.setTimeout(() => {
    elements.toast.classList.remove("show");
  }, 2200);
}

function setBusy(isBusy) {
  elements.runButton.disabled = isBusy;
  elements.runButton.textContent = isBusy ? "Generating CRUMB…" : "Generate CRUMB";
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatCrumb(text) {
  let afterDivider = false;
  return text
    .split("\n")
    .map((line) => {
      const safeLine = escapeHtml(line || " ");
      if (line === "BEGIN CRUMB" || line === "END CRUMB") {
        return `<div class="crumb-line marker">${safeLine}</div>`;
      }
      if (line === "---") {
        afterDivider = true;
        return `<div class="crumb-line divider">${safeLine}</div>`;
      }
      if (!afterDivider && line.includes("=")) {
        const [key, ...rest] = line.split("=");
        return `<div class="crumb-line header"><span class="key">${escapeHtml(key)}</span><span class="equals">=</span><span class="value">${escapeHtml(rest.join("="))}</span></div>`;
      }
      if (/^\[[^\]]+\]$/.test(line.trim())) {
        return `<div class="crumb-line section">${safeLine}</div>`;
      }
      if (/^\s*- /.test(line)) {
        return `<div class="crumb-line bullet">${safeLine}</div>`;
      }
      return `<div class="crumb-line body">${safeLine}</div>`;
    })
    .join("");
}

function renderStats(stats) {
  if (!stats) {
    elements.inputStat.textContent = "0 chars";
    elements.outputStat.textContent = "0 chars";
    elements.ratioStat.textContent = "-";
    return;
  }

  const direction = stats.char_delta >= 0 ? "tighter" : "expanded";
  const ratio = stats.output_ratio ? `${(stats.output_ratio * 100).toFixed(0)}% of input` : "-";
  elements.inputStat.textContent = `${stats.input_chars} chars · ${stats.input_tokens} tok`;
  elements.outputStat.textContent = `${stats.output_chars} chars · ${stats.output_tokens} tok`;
  elements.ratioStat.textContent = `${ratio} · ${direction}`;
}

function applyResult(result) {
  state.currentResult = result;
  elements.mode.value = result.mode;
  elements.modeBadge.textContent = result.mode;
  elements.title.value = result.title || "";
  elements.source.value = result.source || "";
  const stats = result.stats || {};
  elements.outputPreview.innerHTML = formatCrumb(result.outputText);
  elements.outputPreview.classList.remove("hidden");
  elements.outputEmpty.classList.add("hidden");
  elements.resultStrip.classList.remove("hidden");
  elements.copyButton.disabled = false;
  elements.saveButton.disabled = false;
  elements.exportButton.disabled = false;
  elements.statusValue.textContent = "Generated";
  elements.actionNote.textContent = `Ready to paste or save · ${result.mode} output`;
  elements.resultFlow.textContent = `${result.mode.toUpperCase()} · ${result.source || "crumb.studio"}`;
  elements.resultShape.textContent = `${stats.input_lines || 0} raw lines → ${stats.output_lines || 0} structured`;
  elements.resultTargets.textContent = "Claude · ChatGPT · Cursor · Codex";
  renderStats(stats);
}

function clearError() {
  elements.errorBanner.textContent = "";
  elements.errorBanner.classList.add("hidden");
}

function showError(message) {
  elements.errorBanner.textContent = message;
  elements.errorBanner.classList.remove("hidden");
  elements.statusValue.textContent = "Needs attention";
}

function clearResult() {
  state.currentResult = null;
  elements.outputPreview.innerHTML = "";
  elements.outputPreview.classList.add("hidden");
  elements.outputEmpty.classList.remove("hidden");
  elements.resultStrip.classList.add("hidden");
  elements.copyButton.disabled = true;
  elements.saveButton.disabled = true;
  elements.exportButton.disabled = true;
  elements.statusValue.textContent = "Ready";
  elements.actionNote.textContent = "Designed for fast screenshots, demos, and copy-paste handoffs.";
  renderStats(null);
}

async function loadPreset(presetKey, autoRun = true) {
  const preset = DEMO_PRESETS[presetKey];
  if (!preset) return;

  elements.presetCards.forEach((card) => {
    card.classList.toggle("active", card.dataset.preset === presetKey);
  });
  elements.mode.value = preset.mode;
  elements.title.value = preset.title;
  elements.source.value = preset.source;
  elements.input.value = preset.inputText;
  clearError();

  if (autoRun) {
    await runGeneration();
  } else {
    clearResult();
  }
}

function renderHistory(items) {
  state.history = items || [];
  elements.historyList.innerHTML = "";

  if (!state.history.length) {
    elements.historyList.innerHTML = `
      <div class="empty-state">
        <strong>No history yet.</strong>
        <p>Run a few transformations and they will appear here for quick reloads.</p>
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
        stats: entry.stats || null,
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
  showToast("Structured CRUMB output generated.");
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

function wireEvents() {
  elements.runButton.addEventListener("click", runGeneration);
  elements.copyButton.addEventListener("click", copyOutput);
  elements.saveButton.addEventListener("click", saveOutput);
  elements.clearButton.addEventListener("click", () => {
    elements.input.value = "";
    elements.title.value = "";
    clearError();
    clearResult();
  });
  elements.clearInputButton.addEventListener("click", () => {
    elements.input.value = "";
    clearError();
  });
  elements.historyButton.addEventListener("click", () => toggleHistory(true));
  elements.closeHistoryButton.addEventListener("click", () => toggleHistory(false));
  elements.clearHistoryButton.addEventListener("click", clearHistory);
  elements.presetCards.forEach((card) => {
    card.addEventListener("click", async () => {
      await loadPreset(card.dataset.preset, true);
      showToast(`${card.querySelector("strong")?.textContent || "Demo"} loaded.`);
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
  });
}

async function bootstrap() {
  const response = await bridge().bootstrap();
  if (response.ok) {
    state.appVersion = response.appVersion || "dev";
    elements.source.value = response.defaultSource || "crumb.studio";
    renderHistory(response.history || []);
    if (!window.pywebview || !window.pywebview.api) {
      elements.statusValue.textContent = "Preview mode";
      elements.actionNote.textContent = "Bridge unavailable in the browser preview. Run via crumb studio for the real engine.";
    }
    if (!elements.input.value.trim()) {
      await loadPreset("launch-task", !window.pywebview || !window.pywebview.api);
    }
  }
}

wireEvents();

if (window.pywebview) {
  window.addEventListener("pywebviewready", bootstrap);
} else {
  bootstrap();
}
