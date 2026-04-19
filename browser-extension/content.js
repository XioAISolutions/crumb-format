const BUTTON_ID = "crumb-copy-button";
const TOAST_ID = "crumb-toast";
const MAX_VISIBLE_MESSAGES = 4;

const PLATFORM_CONFIG = {
  chatgpt: {
    label: "ChatGPT",
    source: "browser.chatgpt",
    hosts: ["chatgpt.com", "chat.openai.com"],
    containerSelectors: ["main", "[data-testid='conversation-turns']", "#__next main"],
    messageSelectors: [
      { selector: "[data-message-author-role='user']", role: "user" },
      { selector: "[data-message-author-role='assistant']", role: "assistant" },
      { selector: "article[data-testid^='conversation-turn-']", role: "unknown" }
    ]
  },
  claude: {
    label: "Claude",
    source: "browser.claude",
    hosts: ["claude.ai"],
    containerSelectors: ["main", "[data-testid='chat-container']", "section[aria-label='Artifacts']"],
    messageSelectors: [
      { selector: "[data-testid='user-message']", role: "user" },
      { selector: "[data-testid='assistant-message']", role: "assistant" },
      { selector: "[data-testid*='message']", role: "unknown" }
    ]
  },
  gemini: {
    label: "Gemini",
    source: "browser.gemini",
    hosts: ["gemini.google.com"],
    containerSelectors: ["main", "chat-app", "mat-sidenav-content"],
    messageSelectors: [
      { selector: "user-query", role: "user" },
      { selector: "model-response", role: "assistant" },
      { selector: "[data-test-id='user-query']", role: "user" },
      { selector: "[data-test-id='model-response']", role: "assistant" }
    ]
  }
};

let injectTimer = null;
let observing = false;

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "parse-crumb") {
    try {
      sendResponse({ crumb: parseSelectionToCrumb(message.text || "") });
    } catch (error) {
      sendResponse({ error: String(error.message || error) });
    }
    return true;
  }

  if (message.action === "copy-to-clipboard") {
    copyToClipboard(message.text || "")
      .then(() => sendResponse({ ok: true }))
      .catch((error) => sendResponse({ ok: false, error: String(error.message || error) }));
    return true;
  }

  if (message.action === "copy-recent-chat") {
    copyRecentChatAsCrumb()
      .then((crumb) => sendResponse({ ok: true, crumb }))
      .catch((error) => sendResponse({ ok: false, error: String(error.message || error) }));
    return true;
  }

  return false;
});

initialize();

function initialize() {
  if (!getPlatformConfig()) {
    return;
  }
  scheduleButtonInjection(0);
  startObservers();
}

function getPlatformConfig() {
  const hostname = window.location.hostname;
  return Object.values(PLATFORM_CONFIG).find((config) =>
    config.hosts.some((host) => hostname === host || hostname.endsWith(`.${host}`))
  ) || null;
}

function startObservers() {
  if (observing) {
    return;
  }
  observing = true;

  const observer = new MutationObserver(() => scheduleButtonInjection(250));
  observer.observe(document.documentElement || document.body, {
    childList: true,
    subtree: true
  });

  window.addEventListener("popstate", () => scheduleButtonInjection(250));
  window.addEventListener("hashchange", () => scheduleButtonInjection(250));
}

function scheduleButtonInjection(delayMs) {
  clearTimeout(injectTimer);
  injectTimer = setTimeout(() => {
    injectCopyButton();
  }, delayMs);
}

function injectCopyButton() {
  if (document.getElementById(BUTTON_ID)) {
    return;
  }

  const platform = getPlatformConfig();
  if (!platform || !findChatContainer(platform)) {
    return;
  }

  const button = document.createElement("button");
  button.id = BUTTON_ID;
  button.type = "button";
  button.textContent = "Copy as CRUMB";
  button.setAttribute("aria-label", "Copy recent chat as CRUMB");

  Object.assign(button.style, {
    position: "fixed",
    right: "20px",
    bottom: "20px",
    zIndex: "2147483647",
    padding: "10px 14px",
    borderRadius: "999px",
    border: "1px solid rgba(255,255,255,0.12)",
    background: "linear-gradient(135deg, #f97316, #fb923c)",
    color: "#111827",
    fontSize: "13px",
    fontWeight: "700",
    fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    boxShadow: "0 10px 24px rgba(0,0,0,0.22)",
    cursor: "pointer"
  });

  button.addEventListener("mouseenter", () => {
    button.style.transform = "translateY(-1px)";
  });
  button.addEventListener("mouseleave", () => {
    button.style.transform = "translateY(0)";
  });

  button.addEventListener("click", async () => {
    const originalLabel = button.textContent;
    button.disabled = true;
    button.textContent = "Copying…";
    button.style.opacity = "0.85";

    try {
      await copyRecentChatAsCrumb();
    } catch (error) {
      showToast(error.message || "Could not create CRUMB.", true);
    } finally {
      button.disabled = false;
      button.textContent = originalLabel;
      button.style.opacity = "1";
    }
  });

  document.body.appendChild(button);
}

function findChatContainer(platform) {
  for (const selector of platform.containerSelectors) {
    const node = document.querySelector(selector);
    if (node && isVisible(node)) {
      return node;
    }
  }
  return null;
}

async function copyRecentChatAsCrumb() {
  const crumb = buildLogCrumbFromVisibleMessages();
  await copyToClipboard(crumb);
  showToast("Copied as CRUMB.");
  return crumb;
}

function buildLogCrumbFromVisibleMessages() {
  const platform = getPlatformConfig();
  if (!platform) {
    throw new Error("This page is not a supported AI chat surface.");
  }

  const messages = collectRecentVisibleMessages(platform, MAX_VISIBLE_MESSAGES);
  if (messages.length === 0) {
    throw new Error("No recent visible chat messages were found.");
  }

  const createdAt = new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
  const title = sanitizeHeaderValue(`Recent handoff from ${platform.label}`);
  const lines = [
    "BEGIN CRUMB",
    "v=1.2",
    "kind=log",
    `title=${title}`,
    `source=${platform.source}`,
    "url=https://github.com/XioAISolutions/crumb-format",
    "---",
    "[entries]"
  ];

  for (const item of messages) {
    const role = item.role === "assistant" ? "ASSISTANT" : "USER";
    const text = truncateText(item.text, 1600).replace(/\n+/g, " ");
    lines.push(`- [${createdAt}] ${role}: ${text}`);
  }

  lines.push("END CRUMB", "");
  return lines.join("\n");
}

function collectRecentVisibleMessages(platform, limit) {
  const seenNodes = new Set();
  const collected = [];

  for (const descriptor of platform.messageSelectors) {
    const nodes = document.querySelectorAll(descriptor.selector);
    for (const node of nodes) {
      if (seenNodes.has(node) || !isVisible(node)) {
        continue;
      }
      const text = extractText(node);
      if (!text) {
        continue;
      }
      seenNodes.add(node);
      collected.push({
        node,
        role: descriptor.role === "unknown" ? inferRole(node) : descriptor.role,
        text
      });
    }
  }

  if (collected.length === 0) {
    const fallbackNodes = document.querySelectorAll("main article, main [role='listitem'], main section, main div");
    for (const node of fallbackNodes) {
      if (seenNodes.has(node) || !isVisible(node)) {
        continue;
      }
      const text = extractText(node);
      if (!text || text.length < 20) {
        continue;
      }
      seenNodes.add(node);
      collected.push({ node, role: inferRole(node), text });
    }
  }

  const deduped = [];
  const seenKeys = new Set();
  for (const item of collected) {
    const key = `${item.role}::${item.text}`;
    if (seenKeys.has(key)) {
      continue;
    }
    seenKeys.add(key);
    deduped.push(item);
  }

  const trimmed = deduped.slice(-limit);
  return trimmed.map((item) => ({
    role: item.role === "assistant" ? "assistant" : "user",
    text: item.text
  }));
}

function inferRole(node) {
  const attrRole = node.getAttribute("data-message-author-role");
  if (attrRole === "assistant" || attrRole === "user") {
    return attrRole;
  }

  const testId = (node.getAttribute("data-testid") || "").toLowerCase();
  if (testId.includes("assistant") || testId.includes("model")) {
    return "assistant";
  }
  if (testId.includes("user") || testId.includes("human")) {
    return "user";
  }

  const tagName = node.tagName.toLowerCase();
  if (tagName === "model-response") {
    return "assistant";
  }
  if (tagName === "user-query") {
    return "user";
  }

  const className = String(node.className || "").toLowerCase();
  if (className.includes("assistant") || className.includes("markdown") || className.includes("model")) {
    return "assistant";
  }
  if (className.includes("user") || className.includes("human")) {
    return "user";
  }

  return "assistant";
}

function parseSelectionToCrumb(text) {
  const cleaned = normalizeText(text);
  const goal = inferGoal(cleaned);
  const contextItems = buildContextItems(cleaned);
  const constraintItems = inferConstraints(cleaned);
  const handoffActions = inferHandoffActions(cleaned);

  const lines = [
    "BEGIN CRUMB",
    "v=1.2",
    "kind=task",
    "title=Selection handoff",
    "source=browser.selection",
    "url=https://github.com/XioAISolutions/crumb-format",
    "---",
    "[goal]",
    goal,
    "",
    "[context]"
  ];

  contextItems.forEach((item) => lines.push(`- ${item}`));
  lines.push("", "[constraints]");

  if (constraintItems.length === 0) {
    lines.push("- Preserve the intent and details from the selected text.");
  } else {
    constraintItems.forEach((item) => lines.push(`- ${item}`));
  }

  // v1.2 [handoff] — explicit "next AI do this" block.
  lines.push("", "[handoff]");
  if (handoffActions.length > 0) {
    handoffActions.forEach((item) => lines.push(`- to=any  do=${item}`));
  } else {
    lines.push("- to=any  do=act on the goal above using the context bullets");
  }

  lines.push("END CRUMB", "");
  return lines.join("\n");
}

function inferHandoffActions(text) {
  const lines = text.split(/\n+/).map((l) => normalizeText(l)).filter(Boolean);
  const matches = [];
  const seen = new Set();
  const action = /\b(TODO|should|need to|fix|implement|add|ship|write|test)\b/i;
  for (const line of lines) {
    if (line.length < 8) continue;
    if (!action.test(line)) continue;
    const cleaned = line.replace(/^[-*>]+\s*/, "").replace(/^#+\s*/, "").replace(/^\d+\.\s*/, "");
    const key = cleaned.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    matches.push(truncateText(cleaned, 200));
    if (matches.length >= 5) break;
  }
  return matches;
}

function buildContextItems(text) {
  const items = [];
  const lines = text
    .split(/\n+/)
    .map((line) => normalizeText(line))
    .filter(Boolean);

  for (const line of lines) {
    if (line.length < 8) {
      continue;
    }
    items.push(truncateText(line, 220));
    if (items.length >= 6) {
      break;
    }
  }

  if (items.length === 0) {
    items.push("Selected excerpt from an AI conversation.");
  }
  return items;
}

function inferGoal(text) {
  const sentences = text.split(/[.!?\n]/).map((part) => normalizeText(part)).filter((part) => part.length > 12);
  if (sentences.length > 0) {
    return truncateText(sentences[0], 180);
  }
  return "Continue the work described in the selected conversation excerpt.";
}

function inferConstraints(text) {
  const matches = [];
  const seen = new Set();
  const patterns = [
    /(?:must not|cannot|don't|do not|avoid|should not|shouldn't)\s+(.+?)(?:[.;\n]|$)/gim,
    /(?:constraint|requirement|limitation)[\s:]+(.+?)(?:[.;\n]|$)/gim
  ];

  for (const pattern of patterns) {
    let match;
    while ((match = pattern.exec(text)) !== null) {
      const value = normalizeText(match[1] || match[0]);
      const key = value.toLowerCase();
      if (!value || seen.has(key)) {
        continue;
      }
      seen.add(key);
      matches.push(truncateText(value, 180));
    }
  }

  return matches.slice(0, 4);
}

function extractText(node) {
  const text = normalizeText(node.innerText || node.textContent || "");
  if (!text) {
    return "";
  }
  return truncateText(text, 1600);
}

function normalizeText(text) {
  return String(text || "")
    .replace(/\u00a0/g, " ")
    .replace(/\s+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

function sanitizeHeaderValue(value) {
  return String(value || "handoff").replace(/[\r\n]+/g, " ").trim();
}

function truncateText(text, limit) {
  if (text.length <= limit) {
    return text;
  }
  return `${text.slice(0, limit - 1).trimEnd()}…`;
}

function isVisible(node) {
  if (!node || !(node instanceof Element)) {
    return false;
  }
  const style = window.getComputedStyle(node);
  if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") {
    return false;
  }
  const rect = node.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
}

async function copyToClipboard(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
}

function showToast(message, isError = false) {
  const existing = document.getElementById(TOAST_ID);
  if (existing) {
    existing.remove();
  }

  const toast = document.createElement("div");
  toast.id = TOAST_ID;
  toast.textContent = message;

  Object.assign(toast.style, {
    position: "fixed",
    right: "20px",
    bottom: "72px",
    zIndex: "2147483647",
    background: isError ? "#7f1d1d" : "#111827",
    color: "#ffffff",
    borderLeft: `4px solid ${isError ? "#ef4444" : "#f97316"}`,
    borderRadius: "10px",
    padding: "12px 16px",
    maxWidth: "320px",
    boxShadow: "0 10px 24px rgba(0,0,0,0.28)",
    fontSize: "13px",
    fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    opacity: "0",
    transition: "opacity 0.2s ease"
  });

  document.body.appendChild(toast);
  requestAnimationFrame(() => {
    toast.style.opacity = "1";
  });

  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 220);
  }, 2200);
}
