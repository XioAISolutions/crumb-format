// Listen for messages from the background service worker
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "parse-crumb") {
    const crumb = parseToCrumb(message.text);
    sendResponse({ crumb });
  } else if (message.action === "copy-to-clipboard") {
    copyToClipboard(message.text);
  }
  return true; // keep channel open for async response
});

/**
 * Parse raw selected text into a structured CRUMB format.
 */
function parseToCrumb(text) {
  const codeBlocks = extractCodeBlocks(text);
  const decisions = extractDecisions(text);
  const actionItems = extractActionItems(text);
  const goal = inferGoal(text);
  const context = buildContext(text, codeBlocks, decisions);
  const constraints = inferConstraints(text);

  let crumb = `---\nformat: crumb\nversion: 1.0\n---\n\n`;

  // [goal]
  crumb += `[goal]\n${goal}\n\n`;

  // [context]
  crumb += `[context]\n${context}\n\n`;

  // [constraints]
  if (constraints.length > 0) {
    crumb += `[constraints]\n`;
    constraints.forEach((c) => {
      crumb += `- ${c}\n`;
    });
    crumb += `\n`;
  }

  // [decisions]
  if (decisions.length > 0) {
    crumb += `[decisions]\n`;
    decisions.forEach((d) => {
      crumb += `- ${d}\n`;
    });
    crumb += `\n`;
  }

  // [action_items]
  if (actionItems.length > 0) {
    crumb += `[action_items]\n`;
    actionItems.forEach((item) => {
      crumb += `- [ ] ${item}\n`;
    });
    crumb += `\n`;
  }

  // [code_snippets]
  if (codeBlocks.length > 0) {
    crumb += `[code_snippets]\n`;
    codeBlocks.forEach((block, i) => {
      crumb += `\`\`\`${block.lang}\n${block.code}\n\`\`\`\n`;
      if (i < codeBlocks.length - 1) crumb += `\n`;
    });
    crumb += `\n`;
  }

  return crumb.trimEnd() + "\n";
}

/**
 * Extract fenced code blocks (``` ... ```)
 */
function extractCodeBlocks(text) {
  const blocks = [];
  const regex = /```(\w*)\n?([\s\S]*?)```/g;
  let match;
  while ((match = regex.exec(text)) !== null) {
    blocks.push({
      lang: match[1] || "",
      code: match[2].trim()
    });
  }
  return blocks;
}

/**
 * Extract decision statements from the text.
 */
function extractDecisions(text) {
  const patterns = [
    /(?:decided to|going with|let's use|we'll use|choosing|opted for|settled on)\s+(.+?)(?:\.|$)/gim
  ];
  const decisions = [];
  const seen = new Set();

  for (const pattern of patterns) {
    let match;
    while ((match = pattern.exec(text)) !== null) {
      const decision = match[0].trim().replace(/\.$/, "");
      const key = decision.toLowerCase();
      if (!seen.has(key)) {
        seen.add(key);
        decisions.push(capitalizeFirst(decision));
      }
    }
  }
  return decisions;
}

/**
 * Extract action items (TODOs, next steps, etc.)
 */
function extractActionItems(text) {
  const patterns = [
    /(?:TODO|FIXME|HACK)[\s:]+(.+?)(?:\n|$)/gi,
    /(?:next step|need to|needs to|should|must)[\s:]+(.+?)(?:\.|;|\n|$)/gim,
    /(?:fix|implement|add|create|update|refactor|write|build|set up|configure)\s+(.+?)(?:\.|;|\n|$)/gim
  ];
  const items = [];
  const seen = new Set();

  // Specific TODO/FIXME patterns get full match
  let match;
  const todoPattern = /(?:TODO|FIXME|HACK)[\s:]+(.+?)(?:\n|$)/gi;
  while ((match = todoPattern.exec(text)) !== null) {
    const item = match[1].trim();
    const key = item.toLowerCase();
    if (!seen.has(key) && item.length > 3) {
      seen.add(key);
      items.push(capitalizeFirst(item));
    }
  }

  // "next step" / "need to" patterns
  const needPattern = /(?:next step|need to|needs to)[\s:]+(.+?)(?:\.|;|\n|$)/gim;
  while ((match = needPattern.exec(text)) !== null) {
    const item = match[1].trim();
    const key = item.toLowerCase();
    if (!seen.has(key) && item.length > 3) {
      seen.add(key);
      items.push(capitalizeFirst(item));
    }
  }

  return items;
}

/**
 * Infer a goal from the text — uses the first meaningful sentence.
 */
function inferGoal(text) {
  // Strip code blocks for goal inference
  const cleaned = text.replace(/```[\s\S]*?```/g, "").trim();
  const sentences = cleaned.split(/[.\n]/).filter((s) => s.trim().length > 10);
  if (sentences.length > 0) {
    return capitalizeFirst(sentences[0].trim());
  }
  return "Complete the task described in the conversation";
}

/**
 * Build context from the text, code, and decisions.
 */
function buildContext(text, codeBlocks, decisions) {
  const parts = [];

  // Summarize conversation length
  const wordCount = text.split(/\s+/).length;
  parts.push(`Extracted from conversation (~${wordCount} words).`);

  if (codeBlocks.length > 0) {
    const langs = [...new Set(codeBlocks.map((b) => b.lang).filter(Boolean))];
    if (langs.length > 0) {
      parts.push(`Code languages: ${langs.join(", ")}.`);
    }
    parts.push(`${codeBlocks.length} code snippet(s) included.`);
  }

  if (decisions.length > 0) {
    parts.push(`${decisions.length} decision(s) noted.`);
  }

  return parts.join(" ");
}

/**
 * Infer constraints from the text.
 */
function inferConstraints(text) {
  const constraints = [];
  const patterns = [
    /(?:must not|cannot|don't|do not|should not|shouldn't|avoid)\s+(.+?)(?:\.|;|\n|$)/gim,
    /(?:constraint|requirement|limitation)[\s:]+(.+?)(?:\.|;|\n|$)/gim
  ];
  const seen = new Set();

  for (const pattern of patterns) {
    let match;
    while ((match = pattern.exec(text)) !== null) {
      const constraint = match[0].trim().replace(/\.$/, "");
      const key = constraint.toLowerCase();
      if (!seen.has(key) && constraint.length > 5) {
        seen.add(key);
        constraints.push(capitalizeFirst(constraint));
      }
    }
  }
  return constraints;
}

function capitalizeFirst(str) {
  if (!str) return str;
  return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Copy text to clipboard and show toast notification.
 */
function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(
    () => showToast(),
    (err) => {
      // Fallback: textarea copy
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      showToast();
    }
  );
}

/**
 * Show a small toast notification with orange accent.
 */
function showToast() {
  const existing = document.getElementById("crumb-toast");
  if (existing) existing.remove();

  const toast = document.createElement("div");
  toast.id = "crumb-toast";
  toast.textContent = "Crumb copied to clipboard!";
  Object.assign(toast.style, {
    position: "fixed",
    bottom: "24px",
    right: "24px",
    background: "#1e1e1e",
    color: "#fff",
    padding: "12px 20px",
    borderRadius: "8px",
    borderLeft: "4px solid #f97316",
    fontFamily: "'Segoe UI', system-ui, sans-serif",
    fontSize: "14px",
    zIndex: "2147483647",
    boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
    transition: "opacity 0.3s ease",
    opacity: "0"
  });

  document.body.appendChild(toast);

  // Fade in
  requestAnimationFrame(() => {
    toast.style.opacity = "1";
  });

  // Fade out and remove after 2.5s
  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 300);
  }, 2500);
}
