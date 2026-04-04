// Open GitHub link in new tab
document.getElementById("github-link").addEventListener("click", (e) => {
  e.preventDefault();
  chrome.tabs.create({ url: "https://github.com/XioAISolutions/crumb-format" });
});

// Copy pip install command to clipboard
document.getElementById("pip-link").addEventListener("click", (e) => {
  e.preventDefault();
  navigator.clipboard.writeText("pip install crumb-format").then(() => {
    const label = e.currentTarget.querySelector(".label");
    const original = label.textContent;
    label.textContent = "Copied!";
    label.style.color = "#f97316";
    setTimeout(() => {
      label.textContent = original;
      label.style.color = "";
    }, 1500);
  });
});
