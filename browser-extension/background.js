chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "crumb-it",
    title: "Copy as CRUMB",
    contexts: ["selection", "page"],
    documentUrlPatterns: [
      "https://chatgpt.com/*",
      "https://chat.openai.com/*",
      "https://claude.ai/*",
      "https://gemini.google.com/*"
    ]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (!tab || !tab.id || info.menuItemId !== "crumb-it") {
    return;
  }

  if (info.selectionText && info.selectionText.trim()) {
    chrome.tabs.sendMessage(
      tab.id,
      {
        action: "parse-crumb",
        text: info.selectionText
      },
      (response) => {
        if (chrome.runtime.lastError) {
          console.error("CRUMB parse error:", chrome.runtime.lastError.message);
          return;
        }

        if (response && response.crumb) {
          chrome.tabs.sendMessage(tab.id, {
            action: "copy-to-clipboard",
            text: response.crumb
          });
        }
      }
    );
    return;
  }

  chrome.tabs.sendMessage(tab.id, { action: "copy-recent-chat" }, (response) => {
    if (chrome.runtime.lastError) {
      console.error("CRUMB capture error:", chrome.runtime.lastError.message);
      return;
    }
    if (response && response.error) {
      console.error("CRUMB capture error:", response.error);
    }
  });
});
