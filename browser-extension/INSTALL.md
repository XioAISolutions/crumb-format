# Chrome Extension — Quick Install

## Load directly from repo (fastest)

1. Clone the repo: `git clone https://github.com/XioAISolutions/crumb-format.git`
2. Open Chrome → `chrome://extensions`
3. Enable **Developer mode** (top right toggle)
4. Click **Load unpacked**
5. Select the `browser-extension/` folder
6. Done — you'll see the CRUMB icon in your toolbar

## For Chrome Web Store upload

1. Clone the repo
2. Run: `cd browser-extension && zip -r ../crumb-extension.zip . -x generate_icons.py icon.svg README.md INSTALL.md`
3. Upload the zip at https://chrome.google.com/webstore/devconsole
