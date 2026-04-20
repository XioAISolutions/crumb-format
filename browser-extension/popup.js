/**
 * Popup UI for the CRUMB extension — a compact in-browser prompt compressor.
 * Uses the Metalk JS port; nothing is sent to a server.
 */
"use strict";

const STORAGE_KEY = "crumb_ext_v1";
const PRESET_NAMES = { 1: "Safe", 2: "Tighter", 3: "Aggressive", 4: "Skeleton", 5: "Adaptive" };

const $ = (id) => document.getElementById(id);
const input = $("input"), output = $("output");
const hero = $("hero"), heroValue = $("hero-value"), heroSub = $("hero-sub");

// ── State ────────────────────────────────────────────────────

function load() {
  return new Promise((resolve) => {
    if (chrome && chrome.storage && chrome.storage.local) {
      chrome.storage.local.get([STORAGE_KEY], (r) => resolve(r[STORAGE_KEY] || {}));
    } else {
      try { resolve(JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}")); }
      catch (_) { resolve({}); }
    }
  });
}
function save(patch) {
  load().then((current) => {
    const merged = Object.assign({}, current, patch);
    if (chrome && chrome.storage && chrome.storage.local) {
      chrome.storage.local.set({ [STORAGE_KEY]: merged });
    } else {
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(merged)); } catch (_) {}
    }
  });
}

let currentLevel = 2;

function setLevel(lv) {
  currentLevel = lv;
  document.querySelectorAll(".preset").forEach((el) => {
    el.classList.toggle("active", parseInt(el.dataset.level, 10) === lv);
  });
  save({ level: lv });
  compress();
}

document.querySelectorAll(".preset").forEach((el) => {
  el.addEventListener("click", () => setLevel(parseInt(el.dataset.level, 10)));
});

// ── Compression ──────────────────────────────────────────────

let metalkReady = false;

function ensureMetalk() {
  if (metalkReady) return Promise.resolve();
  return self.Metalk.load(chrome.runtime.getURL("metalk-data.json"))
    .then(() => { metalkReady = true; });
}

function estimateTokens(text) { return Math.max(1, Math.floor(text.length / 4)); }

function updateInputStats() {
  $("input-stats").textContent =
    `${input.value.length} chars · ~${estimateTokens(input.value)} tokens`;
}

function setHero(stats) {
  if (!stats) {
    hero.classList.add("empty");
    heroValue.textContent = "—";
    heroSub.textContent = "ready";
    $("output-stats").textContent = "—";
    return;
  }
  hero.classList.remove("empty");
  heroValue.textContent = `${stats.pct_saved}%`;
  heroSub.textContent = `${stats.original_tokens}→${stats.encoded_tokens} tok · L${stats.level} ${stats.mode}`;
  $("output-stats").textContent = `${stats.encoded_chars} chars · ~${stats.encoded_tokens} tokens`;
}

function compressLocal(text, level) {
  // Plain mode calls Metalk.encodePlain directly — no synthetic CRUMB wrap,
  // so user `[goal]` / `[context]` bracket headings are preserved verbatim.
  const isCrumb = text.trim().startsWith("BEGIN CRUMB") || text.trim().startsWith("BC");
  const encoded = isCrumb
    ? self.Metalk.encode(text, level, { vowel_min_length: 4 })
    : self.Metalk.encodePlain(text, level, { vowel_min_length: 4 });
  const stats = self.Metalk.compressionStats(text, encoded);
  stats.level = level;
  stats.mode = isCrumb ? "crumb" : "plain";
  return { encoded, stats };
}

let debounceTimer = null;
function compress() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    const text = input.value;
    if (!text.trim()) { output.value = ""; setHero(null); return; }
    ensureMetalk().then(() => {
      try {
        const res = compressLocal(text, currentLevel);
        output.value = res.encoded;
        setHero(res.stats);
      } catch (err) {
        output.value = "Error: " + err.message;
        setHero(null);
      }
    });
  }, 180);
}

input.addEventListener("input", () => { updateInputStats(); compress(); });

// ── Copy ─────────────────────────────────────────────────────

const copyBtn = $("btn-copy"), copyText = $("btn-copy-text");
let copyReset = null;

function doCopy() {
  if (!output.value) return;
  navigator.clipboard.writeText(output.value).then(() => {
    copyBtn.classList.add("copied");
    copyText.textContent = "✓ Copied!";
    clearTimeout(copyReset);
    copyReset = setTimeout(() => {
      copyBtn.classList.remove("copied");
      copyText.textContent = "Copy compressed";
    }, 1300);
  }).catch(() => {
    output.select();
    document.execCommand("copy");
  });
}

copyBtn.addEventListener("click", doCopy);

$("btn-clear").addEventListener("click", () => {
  input.value = ""; output.value = "";
  updateInputStats(); setHero(null);
  input.focus();
});

// ── Paste & selection pull ───────────────────────────────────

$("btn-paste").addEventListener("click", async () => {
  try {
    const text = await navigator.clipboard.readText();
    if (text) { input.value = text; updateInputStats(); compress(); }
  } catch (err) {
    input.placeholder = "Clipboard blocked. Paste with ⌘/Ctrl+V into the box below.";
    input.focus();
  }
});

$("btn-selection").addEventListener("click", async () => {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !tab.id) return;
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => (window.getSelection ? window.getSelection().toString() : "")
    });
    const sel = (results[0] && results[0].result) || "";
    if (sel) {
      input.value = sel;
      updateInputStats();
      compress();
    } else {
      input.placeholder = "No selection on the current page.";
    }
  } catch (err) {
    console.warn("selection pull failed", err);
  }
});

// ── Keyboard shortcuts ───────────────────────────────────────

document.addEventListener("keydown", (e) => {
  const mod = e.metaKey || e.ctrlKey;
  if (!mod) return;
  if (e.key === "Enter") { e.preventDefault(); doCopy(); }
  else if (e.key >= "1" && e.key <= "5") { e.preventDefault(); setLevel(parseInt(e.key, 10)); }
});

// ── Init ─────────────────────────────────────────────────────

load().then((state) => {
  if (state.level) setLevel(state.level);
  else setLevel(2);
  updateInputStats();
  input.focus();
});
