/**
 * Pure-JS port of MeTalk + vowel-strip (cli/metalk.py + cli/vowelstrip.py).
 *
 * The dict/set tables live in metalk-data.json (exported from the Python
 * source of truth by scripts/export_metalk_data.py). This module loads the
 * JSON once on first use and exposes an API that matches the Python surface.
 *
 * Exports (both browser `window.Metalk` and ES-module form):
 *   encode(text, level, opts?)     — full pipeline, levels 1-5 (5 falls back to 4)
 *   decode(text)                   — reverse Layer 1 dict subs; strip mt=/vs= headers
 *   stripText(text, minLen)        — vowel-strip plain prose
 *   stripLine(line, minLen)        — vowel-strip one line
 *   compressionStats(orig, enc)    — {original_tokens, encoded_tokens, …}
 *   load(url?)                     — manually preload the data JSON
 *
 * Adaptive (level 5) has no in-browser embedding model by default, so it
 * falls back to level 4. The browser extension can override by passing a
 * per-line predicate via opts.keepStrip(line, candidate).
 */
(function (root, factory) {
  "use strict";
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.Metalk = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var DATA = null;
  var DATA_URL_DEFAULT = "metalk-data.json";

  // ── Data loading ────────────────────────────────────────────

  function load(url) {
    var target = url || DATA_URL_DEFAULT;
    return fetch(target).then(function (r) {
      if (!r.ok) throw new Error("metalk-data.json fetch failed: " + r.status);
      return r.json();
    }).then(function (data) {
      setData(data);
      return data;
    });
  }

  function setData(data) {
    DATA = data;
    // Pre-build sorted (longest-first) lists for regex building.
    DATA._abbrevSorted = Object.keys(data.abbrev)
      .sort(function (a, b) { return b.length - a.length; })
      .map(function (k) { return [k, data.abbrev[k]]; });
    DATA._phrasesSorted = Object.keys(data.phrase_rewrites)
      .sort(function (a, b) { return b.length - a.length; })
      .map(function (k) { return [k, data.phrase_rewrites[k]]; });
    DATA._abbrevReverseSorted = Object.keys(data.abbrev)
      .map(function (k) { return [data.abbrev[k], k]; })
      .sort(function (a, b) { return b[0].length - a[0].length; });
    DATA._stripSet = new Set(data.strip_words.map(function (w) { return w.toLowerCase(); }));
    DATA._protectedSet = new Set(data.protected_words.map(function (w) { return w.toLowerCase(); }));
    DATA._punctSet = new Set(data.sentence_punct);
    DATA._opaqueSet = new Set(data.opaque_chars);
    DATA._vowelSet = new Set(data.vowels);
    return DATA;
  }

  function ensureData() {
    if (!DATA) throw new Error("Metalk data not loaded. Call Metalk.load() first.");
  }

  // ── Helpers ─────────────────────────────────────────────────

  function escapeRegExp(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function estimateTokens(text) {
    return Math.max(1, Math.floor(text.length / 4));
  }

  function matchCase(target, sourceFirst) {
    // Preserve capitalization of first character from source.
    if (!target) return target;
    if (sourceFirst && sourceFirst[0].toUpperCase() === sourceFirst[0] && target[0].toLowerCase() === target[0]) {
      return target[0].toUpperCase() + target.slice(1);
    }
    return target;
  }

  // ── Layer 1: dictionary substitution ─────────────────────────

  function applyDict(text, sortedPairs) {
    var out = text;
    sortedPairs.forEach(function (pair) {
      var longForm = pair[0], shortForm = pair[1];
      var re = new RegExp("\\b" + escapeRegExp(longForm) + "\\b", "gi");
      out = out.replace(re, function (m) { return matchCase(shortForm, m); });
    });
    return out;
  }

  function reverseDict(text, sortedPairs) {
    var out = text;
    sortedPairs.forEach(function (pair) {
      var shortForm = pair[0], longForm = pair[1];
      var re = new RegExp("\\b" + escapeRegExp(shortForm) + "\\b", "g");
      out = out.replace(re, function (m) { return matchCase(longForm, m); });
    });
    return out;
  }

  // ── Layer 2: grammar stripping ───────────────────────────────

  function stripGrammar(text) {
    // Phrase rewrites first (longest-first).
    DATA._phrasesSorted.forEach(function (pair) {
      var re = new RegExp(escapeRegExp(pair[0]), "gi");
      text = text.replace(re, pair[1]);
    });
    // Filler words per line, preserving structure.
    return text.split("\n").map(function (line) {
      var words = line.split(/\s+/);
      var kept = [];
      words.forEach(function (w) {
        var clean = w.replace(/^[.,;:!?()[\]"']+|[.,;:!?()[\]"']+$/g, "").toLowerCase();
        if (DATA._stripSet.has(clean)) return;
        kept.push(w);
      });
      return kept.join(" ").replace(/  +/g, " ").replace(/^- +/, "- ").replace(/^\s+|\s+$/g, "");
    }).join("\n");
  }

  // ── Layer 3: aggressive condense ─────────────────────────────

  function condenseAggressive(text) {
    var out = text.split("\n").filter(function (line) {
      var s = line.trim();
      return s !== "-" && s !== "- ";
    }).map(function (line) {
      var s = line.trim();
      if (s.startsWith("-") && s.endsWith(".")) return line.replace(/\.$/, "");
      return line;
    }).join("\n");
    return out.replace(/\n{3,}/g, "\n\n");
  }

  // ── Layer 4/5: vowel strip ──────────────────────────────────

  function shouldStrip(word, minLen) {
    if (word.length < minLen) return false;
    if (word === word.toUpperCase()) return false; // acronym
    if (DATA._protectedSet.has(word.toLowerCase())) return false;
    return true;
  }

  function stripWord(word, minLen) {
    if (minLen === undefined) minLen = 4;
    if (!/^[A-Za-z]+$/.test(word)) return word;
    if (!shouldStrip(word, minLen)) return word;

    var first = word[0];
    var body = word.slice(1);
    var keepS = body.length > 1 && (body.endsWith("s") || body.endsWith("S"));
    var core = keepS ? body.slice(0, -1) : body;
    var stripped = "";
    for (var i = 0; i < core.length; i++) {
      if (!DATA._vowelSet.has(core[i])) stripped += core[i];
    }
    var result = first + stripped + (keepS ? body[body.length - 1] : "");
    if (result.length <= 1 && word.length >= minLen) return word.slice(0, 2);
    return result;
  }

  function isOpaqueToken(tok) {
    for (var i = 0; i < tok.length; i++) {
      if (DATA._opaqueSet.has(tok[i])) return true;
    }
    return false;
  }

  function stripToken(tok, minLen) {
    var leading = "";
    while (tok.length && DATA._punctSet.has(tok[0])) { leading += tok[0]; tok = tok.slice(1); }
    var trailingRev = "";
    while (tok.length && DATA._punctSet.has(tok[tok.length - 1])) {
      trailingRev = tok[tok.length - 1] + trailingRev;
      tok = tok.slice(0, -1);
    }
    if (!tok) return leading + trailingRev;
    var core = isOpaqueToken(tok)
      ? tok
      : tok.replace(/[A-Za-z]+/g, function (m) { return stripWord(m, minLen); });
    return leading + core + trailingRev;
  }

  function stripLine(line, minLen) {
    if (minLen === undefined) minLen = 4;
    ensureData();
    return line.split(/(\s+)/).map(function (part) {
      if (!part || /^\s+$/.test(part)) return part;
      return stripToken(part, minLen);
    }).join("");
  }

  function stripText(text, minLen) {
    return text.split("\n").map(function (l) { return stripLine(l, minLen); }).join("\n");
  }

  // ── CRUMB-aware vowel strip ──────────────────────────────────

  var SECTION_RE = /^\s*\[[^\]]+\]\s*$/;
  var FENCE_RE = /^\s*```/;
  var TYPED_RE = /^\s*@type\s*:\s*(code|diff|json|yaml)\b/i;

  function stripCrumb(text, opts) {
    opts = opts || {};
    var minLen = opts.min_length || 4;
    var transform = opts.transform || function (l) { return stripLine(l, minLen); };

    var lines = text.replace(/\n+$/, "").split("\n");
    var sepIdx = -1;
    for (var i = 0; i < lines.length; i++) {
      if (lines[i].trim() === "---") { sepIdx = i; break; }
    }
    if (sepIdx === -1) return text;

    var out = lines.slice(0, sepIdx + 1);
    var inFence = false;
    var inTypedCode = false;

    for (var j = sepIdx + 1; j < lines.length; j++) {
      var line = lines[j];
      if (SECTION_RE.test(line)) { inTypedCode = false; out.push(line); continue; }
      if (FENCE_RE.test(line)) { inFence = !inFence; out.push(line); continue; }
      var m = TYPED_RE.exec(line);
      if (m) {
        var kind = m[1].toLowerCase();
        inTypedCode = ["code", "diff", "json", "yaml"].indexOf(kind) >= 0;
        out.push(line); continue;
      }
      if (line.trim() === "END CRUMB" || line.trim() === "EC") { out.push(line); continue; }
      if (inFence || inTypedCode) { out.push(line); continue; }
      out.push(transform(line));
    }
    return injectVsHeader(out.join("\n") + "\n", minLen);
  }

  function injectVsHeader(encoded, minLen) {
    var lines = encoded.split("\n");
    var sepIdx = -1;
    for (var i = 0; i < lines.length; i++) if (lines[i].trim() === "---") { sepIdx = i; break; }
    if (sepIdx === -1) return encoded;
    for (var k = 1; k < sepIdx; k++) {
      if (lines[k].trim().startsWith("vs=")) { lines[k] = "vs=" + minLen; return lines.join("\n"); }
    }
    lines.splice(sepIdx, 0, "vs=" + minLen);
    return lines.join("\n");
  }

  // ── Encode / decode pipelines ───────────────────────────────

  function encodePlain(text, level, opts) {
    // Pure body-transform pipeline for arbitrary prose. Mirrors
    // cli/metalk.py::encode_plain — never treats `[bracket]` lines as
    // section headers, so user headings like `[goal]` or `[context]` are
    // preserved verbatim.
    ensureData();
    opts = opts || {};
    level = level || 2;
    var vml = opts.vowel_min_length || 4;
    var result = applyDict(text, DATA._abbrevSorted);
    if (level >= 2) result = stripGrammar(result);
    if (level >= 3) result = condenseAggressive(result);
    if (level >= 4) result = stripText(result, vml);
    return result;
  }

  function encode(text, level, opts) {
    ensureData();
    opts = opts || {};
    level = level || 2;
    var vml = opts.vowel_min_length || 4;

    var lines = text.trim().split("\n");
    var sepIdx = lines.indexOf("---");
    if (sepIdx === -1) {
      // Not a structured crumb — run the plain pipeline.
      return encodePlain(text, level, opts);
    }

    if (lines[0].trim() === "BEGIN CRUMB") lines[0] = "BC";
    if (lines[lines.length - 1].trim() === "END CRUMB") lines[lines.length - 1] = "EC";

    var headerLines = [];
    for (var i = 1; i < sepIdx; i++) {
      var line = lines[i];
      if (line.includes("=")) {
        var idx = line.indexOf("=");
        var key = line.slice(0, idx).trim();
        var val = line.slice(idx + 1);
        if (DATA.header_key_map[key]) line = DATA.header_key_map[key] + "=" + val;
      }
      headerLines.push(line);
    }
    headerLines.push("mt=" + level);

    var bodyLines = [];
    for (var j = sepIdx + 1; j < lines.length - 1; j++) {
      var bl = lines[j];
      var stripped = bl.trim();
      if (stripped.startsWith("[") && stripped.endsWith("]")) {
        var name = stripped.slice(1, -1).trim().toLowerCase();
        if (DATA.section_map[name]) {
          var indent = bl.match(/^\s*/)[0];
          bl = indent + "[" + DATA.section_map[name] + "]";
        }
        bodyLines.push(bl);
        continue;
      }
      bl = applyDict(bl, DATA._abbrevSorted);
      if (level >= 2) bl = stripGrammar(bl);
      bodyLines.push(bl);
    }

    if (level >= 3) {
      var bodyText = bodyLines.join("\n");
      bodyText = condenseAggressive(bodyText);
      bodyLines = bodyText.split("\n");
    }

    var result2 = [lines[0]].concat(headerLines, ["---"], bodyLines, [lines[lines.length - 1]]).join("\n") + "\n";

    if (level >= 4) {
      var transform = function (ln) { return stripLine(ln, vml); };
      if (level >= 5 && typeof opts.keepStrip === "function") {
        transform = function (ln) {
          var cand = stripLine(ln, vml);
          return opts.keepStrip(ln, cand) ? cand : ln;
        };
      }
      result2 = stripCrumb(result2, { min_length: vml, transform: transform });
    }
    return result2;
  }

  function decode(text) {
    ensureData();
    var lines = text.trim().split("\n");
    var hasMt = lines.some(function (l) { return l.trim().startsWith("mt="); });
    if (!hasMt) return text;

    if (lines[0].trim() === "BC") lines[0] = "BEGIN CRUMB";
    if (lines[lines.length - 1].trim() === "EC") lines[lines.length - 1] = "END CRUMB";

    var sepIdx = lines.indexOf("---");
    if (sepIdx === -1) return lines.join("\n") + "\n";

    var headerLines = [];
    for (var i = 1; i < sepIdx; i++) {
      var line = lines[i];
      var stripped = line.trim();
      if (stripped.startsWith("mt=") || stripped.startsWith("vs=")) continue;
      if (line.includes("=")) {
        var idx = line.indexOf("=");
        var key = line.slice(0, idx).trim();
        var val = line.slice(idx + 1);
        for (var k in DATA.header_key_map) {
          if (DATA.header_key_map[k] === key) { line = k + "=" + val; break; }
        }
      }
      headerLines.push(line);
    }

    var bodyLines = [];
    for (var j = sepIdx + 1; j < lines.length - 1; j++) {
      var bl = lines[j];
      var strip = bl.trim();
      if (strip.startsWith("[") && strip.endsWith("]")) {
        var name = strip.slice(1, -1).trim().toLowerCase();
        for (var sk in DATA.section_map) {
          if (DATA.section_map[sk] === name) {
            var indent = bl.match(/^\s*/)[0];
            bl = indent + "[" + sk + "]";
            break;
          }
        }
        bodyLines.push(bl);
        continue;
      }
      bl = reverseDict(bl, DATA._abbrevReverseSorted);
      bodyLines.push(bl);
    }

    return [lines[0]].concat(headerLines, ["---"], bodyLines, [lines[lines.length - 1]]).join("\n") + "\n";
  }

  function compressionStats(original, encoded) {
    var ot = estimateTokens(original), et = estimateTokens(encoded);
    var saved = ot - et;
    return {
      original_tokens: ot,
      encoded_tokens: et,
      saved_tokens: saved,
      pct_saved: Math.round((saved / ot) * 1000) / 10,
      ratio: Math.round((ot / Math.max(et, 1)) * 100) / 100,
      original_chars: original.length,
      encoded_chars: encoded.length
    };
  }

  return {
    load: load,
    setData: setData,
    encode: encode,
    encodePlain: encodePlain,
    decode: decode,
    stripText: stripText,
    stripLine: stripLine,
    stripWord: stripWord,
    compressionStats: compressionStats,
    estimateTokens: estimateTokens
  };
});
