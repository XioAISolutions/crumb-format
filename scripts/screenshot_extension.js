/**
 * Screenshot the extension popup by loading popup.html in Chromium with a
 * mock `chrome` API stub. Produces docs/assets/extension-popup.png.
 */
"use strict";

const path = require("path");
const fs = require("fs");
const { chromium } = require("/opt/node22/lib/node_modules/playwright");

const ROOT = path.resolve(__dirname, "..");
const EXT = path.join(ROOT, "browser-extension");
const OUT = path.join(ROOT, "docs", "assets");
fs.mkdirSync(OUT, { recursive: true });

(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({
    viewport: { width: 480, height: 720 }
  });
  const page = await ctx.newPage();

  const BASE = process.env.EXT_BASE || "http://127.0.0.1:18426";

  // Mock the chrome.* APIs the popup uses.
  await page.addInitScript(({ dataUrl }) => {
    window.chrome = {
      runtime: { getURL: () => dataUrl },
      storage: {
        local: {
          get: (keys, cb) => cb({}),
          set: (obj, cb) => cb && cb()
        }
      },
      tabs: { query: async () => [{ id: 1 }] },
      scripting: { executeScript: async () => [{ result: "" }] }
    };
  }, { dataUrl: BASE + "/metalk-data.json" });

  await page.goto(BASE + "/popup.html");
  await page.waitForSelector(".preset.active");
  await page.waitForTimeout(200);

  // Type a sample prompt and wait for compression.
  await page.fill("#input",
    "Please help me fix a bug in the authentication middleware. The application is not properly validating the JSON Web Token when users refresh the page.");
  await page.waitForFunction(() => {
    const h = document.getElementById("hero-value");
    return h && h.textContent.trim() !== "—";
  }, { timeout: 5000 });
  await page.waitForTimeout(200);

  const outFile = path.join(OUT, "extension-popup.png");
  await page.screenshot({ path: outFile, fullPage: true });
  console.log("wrote", outFile);

  // Skeleton preset
  await page.click('.preset[data-level="4"]');
  await page.waitForTimeout(350);
  const outFile2 = path.join(OUT, "extension-popup-skeleton.png");
  await page.screenshot({ path: outFile2, fullPage: true });
  console.log("wrote", outFile2);

  await browser.close();
})().catch((err) => { console.error(err); process.exit(1); });
