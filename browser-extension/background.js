// Create context menu item on install
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "crumb-it",
    title: "Crumb it",
    contexts: ["selection"]
  });
});

// Handle context menu click
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "crumb-it" && info.selectionText) {
    // Send selected text to content script for parsing
    chrome.tabs.sendMessage(
      tab.id,
      {
        action: "parse-crumb",
        text: info.selectionText
      },
      (response) => {
        if (chrome.runtime.lastError) {
          console.error("Crumb error:", chrome.runtime.lastError.message);
          return;
        }
        if (response && response.crumb) {
          // Copy the crumb to clipboard via the content script
          chrome.tabs.sendMessage(tab.id, {
            action: "copy-to-clipboard",
            text: response.crumb
          });
        }
      }
    );
  }
});
