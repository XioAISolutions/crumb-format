/**
 * Playground screenshot harness.
 *
 * Captures a handful of UI states against a live `crumb playground` server
 * (default http://127.0.0.1:18422) and writes PNGs to docs/assets/. Clears
 * localStorage between runs so preset/advanced state is reproducible.
 *
 * Usage:
 *   node scripts/screenshot_playground.js [--base http://127.0.0.1:PORT]
 */
"use strict";

const path = require("path");
const fs = require("fs");
const { chromium } = require("/opt/node22/lib/node_modules/playwright");

const ARGS = process.argv.slice(2);
function arg(name, fallback) {
  const i = ARGS.indexOf(name);
  return i >= 0 ? ARGS[i + 1] : fallback;
}

const BASE = arg("--base", "http://127.0.0.1:18422");
const OUT = path.resolve(__dirname, "..", "docs", "assets");
fs.mkdirSync(OUT, { recursive: true });

const DESKTOP = { width: 1280, height: 860 };
const MOBILE = { width: 420, height: 860 };

async function resetStorage(page) {
  await page.evaluate(() => {
    try { localStorage.clear(); } catch (_) {}
  });
}

async function snap(page, name) {
  const file = path.join(OUT, name);
  await page.screenshot({ path: file, fullPage: false });
  console.log("wrote", file);
}

(async () => {
  const browser = await chromium.launch();

  // ── 1. Empty state, desktop ──
  let ctx = await browser.newContext({ viewport: DESKTOP });
  let page = await ctx.newPage();
  await page.goto(BASE + "/playground.html");
  await resetStorage(page);
  await page.reload();
  await page.waitForSelector(".preset.active");
  await page.waitForTimeout(150);
  await snap(page, "playground-empty.png");
  await ctx.close();

  // ── 2. Example loaded, default preset (Tighter = L2) ──
  ctx = await browser.newContext({ viewport: DESKTOP });
  page = await ctx.newPage();
  await page.goto(BASE + "/playground.html");
  await resetStorage(page);
  await page.reload();
  await page.waitForSelector(".preset.active");
  await page.selectOption("#example", "bug");
  // Wait for debounced compression + hero render
  await page.waitForFunction(() => {
    const h = document.getElementById("hero-value");
    return h && h.textContent.trim() !== "" && h.textContent.trim() !== "—";
  }, { timeout: 5000 });
  await page.waitForTimeout(250);
  await snap(page, "playground-compressed-l2.png");

  // ── 3. Skeleton (L4) on same input ──
  await page.click('.preset[data-level="4"]');
  await page.waitForTimeout(450);
  await snap(page, "playground-skeleton-l4.png");

  // ── 4. Advanced panel open with .crumb example ──
  await page.selectOption("#example", "crumb");
  await page.waitForTimeout(350);
  await page.click("#btn-advanced");
  await page.waitForSelector("#advanced.open");
  await page.waitForTimeout(200);
  await snap(page, "playground-advanced-crumb.png");
  await ctx.close();

  // ── 5. Mobile viewport, example loaded ──
  ctx = await browser.newContext({ viewport: MOBILE, deviceScaleFactor: 2 });
  page = await ctx.newPage();
  await page.goto(BASE + "/playground.html");
  await resetStorage(page);
  await page.reload();
  await page.waitForSelector(".preset.active");
  await page.selectOption("#example", "long");
  await page.waitForFunction(() => {
    const h = document.getElementById("hero-value");
    return h && h.textContent.trim() !== "" && h.textContent.trim() !== "—";
  }, { timeout: 5000 });
  await page.waitForTimeout(250);
  await snap(page, "playground-mobile.png");
  await ctx.close();

  await browser.close();
  console.log("done");
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
