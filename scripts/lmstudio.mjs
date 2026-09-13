#!/usr/bin/env node
/**
 * lmstudio.mjs — SEC extraction eval against an LM Studio server.
 *
 * Node 18+ (no npm dependencies). Mirrors the extract-full task from
 * llama-cpp-spark as closely as a single JS file can: same companies, forms
 * (1x 10-K + 3x 10-Q), XBRL ground truth selection, prompts, free-text scoring,
 * and full -> section -> BM25 top-k context degradation.
 *
 *   node scripts/lmstudio.mjs fetch
 *   node scripts/lmstudio.mjs run [--limit N] [--ticker AAPL] [--tag Assets] [--forms 10-K]
 *   node scripts/lmstudio.mjs report
 *
 * Env:
 *   EDGAR_UA    required, e.g. "local-llm you@example.com"  (SEC fair access)
 *   LMS_URL     default http://127.0.0.1:1234
 *   LMS_MODEL   default: first loaded llm from /api/v0/models
 *   LMS_TOKEN   optional bearer token if Require Authentication is on
 *
 * Flags:
 *   --insecure           disable TLS verification (corporate SSL inspection)
 *   --reasoning-effort   low|medium|high (default: unset, matches evals.toml)
 *
 * Config (companies, tags, tolerance, decoding) comes from evals/generated/sec-config.json and
 * the prompts from evals/prompts/*.md, both owned by the Python harness. Blocks marked
 * "Port of ..." re-implement the named Python function; keep them in step when it changes.
 *
 * Known gaps vs the Python harness (not fixable without a tokenizer endpoint):
 *   - Token counts are chars/4.6 estimates; mode boundaries can differ slightly.
 *   - Backend is LM Studio's runtime, not the pinned llama.cpp build; perf
 *     columns are indicative only; cached prompt tokens are unavailable.
 *
 * Re-run `fetch` after upgrading this script: old state/filings used a weaker
 * HTML stripper and a 10-K-only manifest.
 */

import { readFile, writeFile, mkdir, readdir, appendFile, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import http from 'node:http';
import https from 'node:https';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import os from 'node:os';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

// ---------------------------------------------------------------- config
// Everything below is read from files the Python harness owns, never hand-copied:
//   evals/generated/sec-config.json  <- `uv run local-llm eval sec export-config` (from evals.toml)
//   evals/prompts/sec_extract_{system,user}.md
// A Python test asserts the generated JSON matches evals.toml, so this script cannot drift.

const CONFIG_PATH = path.join(ROOT, 'evals', 'generated', 'sec-config.json');
if (!existsSync(CONFIG_PATH)) {
  console.error(`missing ${CONFIG_PATH}; on a Python box run: uv run local-llm eval sec export-config`);
  process.exit(1);
}
const CONFIG = JSON.parse(await readFile(CONFIG_PATH, 'utf8'));
const COMPANIES = CONFIG.companies;              // [{ticker, cik}]
const FORMS = CONFIG.forms;                      // {'10-K': 1, '10-Q': 3}
const TAGS = CONFIG.tags.map((t) => ({ ...t, aliases: t.aliases ?? [], accept_aliases: !!t.accept_aliases }));
const TOLERANCE = CONFIG.tolerance;
const MAX_TOKENS = CONFIG.quality.max_tokens;
const SEED = CONFIG.quality.seed;
const CHUNK_TOKENS = CONFIG.chunk_tokens;
const TOP_K = CONFIG.top_k;

const CHARS_PER_TOKEN = 4.6;      // approximation; LM Studio has no tokenize endpoint
const CHUNK_CHARS = Math.round(CHUNK_TOKENS * CHARS_PER_TOKEN);
const PROMPT_OVERHEAD_TOKENS = 256; // matches spark_llm.evals.sec.run.PROMPT_OVERHEAD_TOKENS

const PROMPTS = path.join(ROOT, 'evals', 'prompts');
const SYSTEM_PROMPT = (await readFile(path.join(PROMPTS, 'sec_extract_system.md'), 'utf8')).trim();
const USER_PROMPT_TEMPLATE = (await readFile(path.join(PROMPTS, 'sec_extract_user.md'), 'utf8')).trim();

const STATE = path.join(ROOT, 'state');
const FILINGS = path.join(STATE, 'filings');
// Same suite tree as the Python harness: state/evals/sec/<model-or-stack>/...
const EVALS = path.join(STATE, 'evals', 'sec');

const argv = process.argv.slice(2);
const CMD = argv[0];
const flag = (name, def = null) => {
  const i = argv.indexOf(`--${name}`);
  return i === -1 ? def : (argv[i + 1]?.startsWith('--') ? true : argv[i + 1]);
};
const has = (name) => argv.includes(`--${name}`);

if (has('insecure')) process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

const UA = process.env.EDGAR_UA || '';
const LMS_URL = (process.env.LMS_URL || 'http://127.0.0.1:1234').replace(/\/$/, '');
const LMS_TOKEN = process.env.LMS_TOKEN || '';
const REASONING_EFFORT = flag('reasoning-effort'); // unset = model default

// ---------------------------------------------------------------- helpers

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const estTokens = (s) => Math.ceil((s?.length ?? 0) / CHARS_PER_TOKEN);
const hash = (o) => createHash('sha256').update(JSON.stringify(o)).digest('hex').slice(0, 12);
const pct = (n, d) => (d === 0 ? '-' : (n / d).toFixed(3));
const quantile = (xs, q) => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(s.length * q))];
};
const p50 = (xs) => quantile(xs, 0.5);
const p95 = (xs) => quantile(xs, 0.95);

async function edgarGet(url, asJson = true) {
  if (!UA) throw new Error('EDGAR_UA is not set. SEC requires a descriptive User-Agent.');
  for (let attempt = 0; attempt < 5; attempt++) {
    const res = await fetch(url, {
      headers: { 'User-Agent': UA, 'Accept-Encoding': 'gzip, deflate' },
    });
    if (res.status === 429 || res.status >= 500) { await sleep(2000 * (attempt + 1)); continue; }
    if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
    await sleep(150);                       // EDGAR fair-access rate limit
    return asJson ? res.json() : res.text();
  }
  throw new Error(`giving up on ${url}`);
}

// ---------------------------------------------------------------- html -> text
// Closer to spark_llm.evals.sec.parse.html_to_text: drop ix:header / display:none,
// flatten tables to pipe rows. Pure regex (no DOM deps).

function flattenTable(tableHtml) {
  const rows = [];
  const trs = tableHtml.match(/<tr[\s\S]*?<\/tr>/gi) ?? [];
  for (const tr of trs) {
    const cells = [...tr.matchAll(/<t[dh][^>]*>([\s\S]*?)<\/t[dh]>/gi)]
      .map((m) => m[1].replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim())
      .filter(Boolean);
    if (!cells.length) continue;
    let line = cells.join(' | ');
    line = line.replace(/\$\s*\|\s*/g, '$')
               .replace(/\(\s*\|\s*/g, '(')
               .replace(/\s*\|\s*\)/g, ')')
               .replace(/\s*\|\s*%/g, '%');
    rows.push(line);
  }
  return rows.join('\n');
}

function htmlToText(html) {
  let s = html;
  // Drop hidden inline-XBRL fact dump and scripts/styles.
  s = s.replace(/<ix:header[\s\S]*?<\/ix:header>/gi, ' ');
  s = s.replace(/<(script|style)[\s\S]*?<\/\1>/gi, ' ');
  s = s.replace(/<!--[\s\S]*?-->/g, ' ');
  // Elements with display:none (nested-div scan via non-greedy tag match).
  s = s.replace(/<([a-z][a-z0-9]*)\b[^>]*style\s*=\s*["'][^"']*display\s*:\s*none[^"']*["'][^>]*>[\s\S]*?<\/\1>/gi, ' ');
  // Flatten tables before stripping remaining tags.
  s = s.replace(/<table[\s\S]*?<\/table>/gi, (t) => '\n' + flattenTable(t) + '\n');
  s = s.replace(/<(br|\/p|\/div|\/tr|\/h[1-6])\s*\/?>/gi, '\n');
  s = s.replace(/<[^>]+>/g, ' ');
  s = s.replace(/&nbsp;|&#160;/gi, ' ')
       .replace(/&amp;/gi, '&').replace(/&lt;/gi, '<').replace(/&gt;/gi, '>')
       .replace(/&#(\d+);/g, (_, d) => String.fromCharCode(+d))
       .replace(/&[a-z]+;/gi, ' ');
  s = s.replace(/[ \t\u00a0\u2009\u202f]+/g, ' ');
  s = s.split('\n').map((ln) => ln.trim()).join('\n');
  s = s.replace(/\n{3,}/g, '\n\n');
  return s.trim() + '\n';
}

// ---------------------------------------------------------------- sections
// Port of find_sections: longest span per Item label beats the TOC.
// 10-Q items are prefixed with the Part (I.1, II.1A, ...).

const FINANCIAL_STATEMENTS = { '10-K': '8', '10-Q': 'I.1' };

function findSections(text, form) {
  const parts = [];
  if (form === '10-Q') {
    for (const m of text.matchAll(/^\s*part\s+(i{1,3}|iv)\b/gim)) {
      parts.push({ pos: m.index, name: m[1].toUpperCase() });
    }
  }
  const heads = [];
  for (const m of text.matchAll(/^\s*item\s+(\d{1,2}[abc]?)\s*[.:\-\u2013\u2014]?\s*(?=\S|$)/gim)) {
    let label = m[1].toUpperCase();
    if (form === '10-Q') {
      let part = 'I';
      for (const p of parts) {
        if (p.pos <= m.index) part = p.name;
      }
      label = `${part}.${label}`;
    }
    heads.push({ pos: m.index, label });
  }
  heads.push({ pos: text.length, label: '__end__' });

  const best = {};
  for (let i = 0; i < heads.length - 1; i++) {
    const { pos, label } = heads[i];
    if (label === '__end__') continue;
    const nxt = heads[i + 1].pos;
    const span = nxt - pos;
    if (!best[label] || span > (best[label].end - best[label].start)) {
      best[label] = { label, start: pos, end: nxt };
    }
  }
  return best;
}

function sectionText(text, form, which) {
  const label = which[form];
  if (!label) return null;
  const sec = findSections(text, form)[label];
  return sec ? text.slice(sec.start, sec.end) : null;
}

// ---------------------------------------------------------------- BM25
// ATIRE Okapi BM25 matching rank_bm25.BM25Okapi (k1=1.5, b=0.75, epsilon=0.25).

function words(s) {
  return s.toLowerCase().match(/[a-z0-9]+(?:'[a-z]+)?/g) ?? [];
}

function bm25TopChunks(text, query, budgetTokens) {
  const chunks = [];
  for (let i = 0; i < text.length; i += CHUNK_CHARS) {
    const c = text.slice(i, i + CHUNK_CHARS);
    if (c) chunks.push(c);
  }
  if (!chunks.length) return '';

  const docs = chunks.map(words);
  const docLens = docs.map((d) => d.length);
  const avgdl = docLens.reduce((a, n) => a + n, 0) / docs.length;
  const N = docs.length;
  const df = new Map();
  const docFreqs = docs.map((d) => {
    const tf = new Map();
    for (const t of d) tf.set(t, (tf.get(t) ?? 0) + 1);
    for (const t of tf.keys()) df.set(t, (df.get(t) ?? 0) + 1);
    return tf;
  });

  // IDF with epsilon floor for negative values (rank_bm25 ATIRE).
  const idf = new Map();
  let idfSum = 0;
  const negative = [];
  for (const [word, freq] of df) {
    const v = Math.log(N - freq + 0.5) - Math.log(freq + 0.5);
    idf.set(word, v);
    idfSum += v;
    if (v < 0) negative.push(word);
  }
  const avgIdf = idf.size ? idfSum / idf.size : 0;
  const eps = 0.25 * avgIdf;
  for (const w of negative) idf.set(w, eps);

  const k1 = 1.5, b = 0.75;
  const q = words(query);
  const scored = docs.map((d, i) => {
    const tf = docFreqs[i];
    let score = 0;
    for (const t of q) {
      const f = tf.get(t) ?? 0;
      if (!f) continue;
      const idfV = idf.get(t) ?? 0;
      score += idfV * (f * (k1 + 1)) / (f + k1 * (1 - b + b * docLens[i] / avgdl));
    }
    return { i, score };
  }).sort((a, b2) => b2.score - a.score);

  // Top-k, then fill budget (harness: take top_k then drop those that blow budget).
  const ranked = scored.slice(0, TOP_K).map((x) => x.i);
  const keep = [];
  let used = 0;
  for (const i of ranked) {
    const t = CHUNK_TOKENS; // harness uses fixed chunk_tokens for budget accounting
    if (budgetTokens != null && used + t > budgetTokens) continue;
    keep.push(i);
    used += t;
  }
  keep.sort((a, b2) => a - b2);
  return keep.map((i) => chunks[i]).join('\n\n[...]\n\n');
}

// ---------------------------------------------------------------- XBRL ground truth
// Port of spark_llm.evals.sec.questions.{_entries,_duration_ok,select_fact,...}.

function daysBetween(start, end) {
  return (Date.parse(end) - Date.parse(start)) / 86400000;
}

function durationOk(kind, form, e) {
  if (kind === 'instant') return !('start' in e) || e.start == null;
  if (!e.start) return false;
  const d = daysBetween(e.start, e.end);
  if (kind === 'duration') {
    return form === '10-K' ? (d >= 340 && d <= 380) : (d >= 80 && d <= 100);
  }
  if (kind === 'ytd') {
    return form === '10-K' ? (d >= 340 && d <= 380) : (d >= 80 && d <= 290);
  }
  return false;
}

function entriesForTag(facts, tag) {
  const out = [];
  const gaap = facts?.facts?.['us-gaap'] ?? {};
  const dei = facts?.facts?.dei ?? {};
  for (const concept of [tag.tag, ...tag.aliases]) {
    const node = gaap[concept] || dei[concept];
    if (!node) continue;
    for (const [unit, series] of Object.entries(node.units ?? {})) {
      if (unit !== tag.unit) continue;
      for (const e of series) out.push({ ...e, concept, unit });
    }
  }
  return out;
}

function periodCandidates(facts, tag, form, reportDate) {
  return entriesForTag(facts, tag).filter(
    (e) => e.end === reportDate && durationOk(tag.kind, form, e),
  );
}

function selectFact(facts, tag, form, accession, reportDate) {
  const cands = periodCandidates(facts, tag, form, reportDate);
  if (!cands.length) return null;
  const own = cands.filter((e) => e.accn === accession);
  const sameForm = cands.filter((e) => e.form === form);
  const pool = own.length ? own : (sameForm.length ? sameForm : cands);
  const order = [tag.tag, ...tag.aliases];

  // Match Python max(key=(-order.index, dur if ytd, "frame" in e, filed)).
  let best = null;
  let bestKey = null;
  for (const e of pool) {
    const idx = order.indexOf(e.concept);
    const dur = e.start ? daysBetween(e.start, e.end) : 0;
    const key = [
      -idx,
      tag.kind === 'ytd' ? dur : 0,
      Object.prototype.hasOwnProperty.call(e, 'frame') ? 1 : 0,
      e.filed ?? '',
    ];
    if (!bestKey || key[0] > bestKey[0]
        || (key[0] === bestKey[0] && key[1] > bestKey[1])
        || (key[0] === bestKey[0] && key[1] === bestKey[1] && key[2] > bestKey[2])
        || (key[0] === bestKey[0] && key[1] === bestKey[1] && key[2] === bestKey[2]
            && String(key[3]) > String(bestKey[3]))) {
      best = e;
      bestKey = key;
    }
  }
  return best;
}

function alternateValues(facts, tag, form, reportDate, chosen) {
  if (!tag.accept_aliases) return [];
  const out = [];
  for (const e of periodCandidates(facts, tag, form, reportDate)) {
    if (e.concept === chosen.concept) continue;
    const v = Number(e.val);
    if (v !== Number(chosen.val) && !out.includes(v)) out.push(v);
  }
  return out;
}

function fmtDate(d) {
  // "September 28, 2025" — matches Python strftime("%B %d, %Y").replace(" 0", " ")
  const dt = new Date(d + 'T00:00:00Z');
  const months = ['January','February','March','April','May','June',
                  'July','August','September','October','November','December'];
  return `${months[dt.getUTCMonth()]} ${dt.getUTCDate()}, ${dt.getUTCFullYear()}`;
}

function periodPhrase(kind, form, e) {
  const end = fmtDate(e.end);
  if (kind === 'instant') return `as of ${end}`;
  if (form === '10-K') return `for the fiscal year ended ${end}`;
  const months = Math.round(daysBetween(e.start, e.end) / 30.4);
  return `for the ${months}-month period ended ${end}`;
}

function formatHint(unit) {
  if (unit === 'USD/shares') {
    return 'Give dollars per share as a plain decimal, e.g. 1.23 (negative as -1.23).';
  }
  if (unit === 'shares') {
    return 'Give the full number of shares, e.g. 1234567890 (not in thousands or millions).';
  }
  return 'Give the full amount in US dollars, e.g. 1234567890 for $1.23 billion '
       + '(not in thousands or millions; negative as -1234567890).';
}

function questionsForFiling(facts, form, accession, reportDate, key) {
  const out = [];
  for (const tag of TAGS) {
    const e = selectFact(facts, tag, form, accession, reportDate);
    if (!e) continue;
    const phrase = periodPhrase(tag.kind, form, e);
    out.push({
      id: `${key}:${tag.tag}`,
      tag: tag.tag,
      label: tag.label,
      kind: tag.kind,
      unit: tag.unit,
      period_start: e.start ?? null,
      period_end: e.end,
      expected: Number(e.val),
      text: `What was ${tag.label} ${phrase}?`,
      format_hint: formatHint(tag.unit),
      concept: e.concept,
      alternates: alternateValues(facts, tag, form, reportDate, e),
    });
  }
  return out;
}

// ---------------------------------------------------------------- fetch

function pickRecent(subs, forms) {
  const r = subs.filings?.recent ?? {};
  const n = (r.form ?? []).length;
  const rows = [];
  for (let i = 0; i < n; i++) {
    rows.push({
      form: r.form[i],
      accessionNumber: r.accessionNumber[i],
      primaryDocument: r.primaryDocument[i],
      filingDate: r.filingDate[i],
      reportDate: r.reportDate[i],
    });
  }
  const picked = [];
  for (const [form, count] of Object.entries(forms)) {
    picked.push(...rows.filter((x) => x.form === form && x.primaryDocument).slice(0, count));
  }
  return picked;
}

async function cmdFetch() {
  await mkdir(FILINGS, { recursive: true });
  const manifest = { created: new Date().toISOString(), filings: [], facts_paths: {} };

  for (const { ticker, cik } of COMPANIES) {
    const cikPad = String(cik).padStart(10, '0');
    process.stdout.write(`${ticker} (CIK ${cikPad}) ... `);

    const cdir = path.join(FILINGS, ticker);
    await mkdir(cdir, { recursive: true });

    const factsPath = path.join(cdir, 'companyfacts.json');
    let facts;
    if (existsSync(factsPath)) {
      facts = JSON.parse(await readFile(factsPath, 'utf8'));
    } else {
      facts = await edgarGet(`https://data.sec.gov/api/xbrl/companyfacts/CIK${cikPad}.json`);
      await writeFile(factsPath, JSON.stringify(facts));
    }
    manifest.facts_paths[ticker] = factsPath;

    const subs = await edgarGet(`https://data.sec.gov/submissions/CIK${cikPad}.json`);
    const company = subs.name ?? ticker;
    const rows = pickRecent(subs, FORMS);
    let nTags = 0, nFilings = 0;

    for (const row of rows) {
      const acc = row.accessionNumber;
      const bare = acc.replace(/-/g, '');
      const url = `https://www.sec.gov/Archives/edgar/data/${cik}/${bare}/${row.primaryDocument}`;
      const htmlPath = path.join(cdir, `${acc}.htm`);
      const textPath = path.join(cdir, `${acc}.txt`);

      if (!existsSync(htmlPath)) {
        const html = await edgarGet(url, false);
        await writeFile(htmlPath, html);
      }
      if (!existsSync(textPath)) {
        const html = await readFile(htmlPath, 'utf8');
        await writeFile(textPath, htmlToText(html));
      }
      const text = await readFile(textPath, 'utf8');
      const key = `${ticker}:${row.form}:${row.reportDate}`;
      const qs = questionsForFiling(facts, row.form, acc, row.reportDate, key);
      nTags += qs.length;
      nFilings++;

      manifest.filings.push({
        ticker, cik, company,
        form: row.form,
        accession: acc,
        primary_document: row.primaryDocument,
        filing_date: row.filingDate,
        report_date: row.reportDate,
        html_path: htmlPath,
        text_path: textPath,
        n_chars: text.length,
        est_tokens: estTokens(text),
        key,
        // Precompute questions so `run` doesn't need companyfacts again.
        questions: qs,
      });
    }
    console.log(`${nFilings} filings, ${nTags} scoreable items`);
  }

  await writeFile(path.join(FILINGS, 'manifest.json'), JSON.stringify(manifest, null, 2));
  const items = manifest.filings.reduce((a, f) => a + f.questions.length, 0);
  console.log(`\n${manifest.filings.length} filings, ${items} scoreable items -> ${FILINGS}/manifest.json`);
}

// ---------------------------------------------------------------- LM Studio

async function lmsModels() {
  let res;
  try {
    res = await fetch(`${LMS_URL}/api/v0/models`, {
      headers: LMS_TOKEN ? { Authorization: `Bearer ${LMS_TOKEN}` } : {},
    });
  } catch (err) {
    throw new Error(
      `Cannot reach LM Studio at ${LMS_URL} (${err.cause?.code ?? err.message}). `
      + 'Start the server or set LMS_URL.',
    );
  }
  if (!res.ok) throw new Error(`LM Studio /api/v0/models -> ${res.status}`);
  return (await res.json()).data;
}

async function lmsProbe(model) {
  try {
    const res = await fetch(`${LMS_URL}/api/v0/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(LMS_TOKEN ? { Authorization: `Bearer ${LMS_TOKEN}` } : {}),
      },
      body: JSON.stringify({
        model, messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 1, temperature: 0, stream: false,
      }),
    });
    if (!res.ok) return null;
    const j = await res.json();
    return { runtime: j.runtime, model_info: j.model_info };
  } catch { return null; }
}

/**
 * Stream via node:http — undici's 300s headersTimeout cannot be overridden and
 * LM Studio withholds headers until prefill finishes (minutes for ~90K tokens).
 */
function lmsStream(model, messages, onFirstToken) {
  const url = new URL(`${LMS_URL}/api/v0/chat/completions`);
  const lib = url.protocol === 'https:' ? https : http;
  const body = {
    model, messages,
    temperature: 0,
    seed: SEED,
    max_tokens: MAX_TOKENS,
    stream: true,
    stream_options: { include_usage: true },
  };
  if (REASONING_EFFORT && REASONING_EFFORT !== true) {
    body.reasoning_effort = REASONING_EFFORT;
  }
  const payload = JSON.stringify(body);

  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    const req = lib.request({
      hostname: url.hostname,
      port: url.port || (url.protocol === 'https:' ? 443 : 80),
      path: url.pathname,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(payload),
        Accept: 'text/event-stream',
        ...(LMS_TOKEN ? { Authorization: `Bearer ${LMS_TOKEN}` } : {}),
      },
    }, (res) => {
      if (res.statusCode !== 200) {
        let errBody = '';
        res.on('data', (d) => { errBody += d; });
        res.on('end', () => reject(new Error(
          `chat/completions -> ${res.statusCode} ${errBody.slice(0, 300)}`)));
        return;
      }

      let buf = '';
      let content = '', reasoning = '';
      let firstTokenAt = null, finishReason = null;
      let usage = null, stats = null, modelInfo = null, runtime = null;

      res.setEncoding('utf8');
      res.on('data', (chunk) => {
        buf += chunk;
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          const t = line.trim();
          if (!t.startsWith('data:')) continue;
          const raw = t.slice(5).trim();
          if (raw === '[DONE]') continue;
          let ev;
          try { ev = JSON.parse(raw); } catch { continue; }

          const delta = ev.choices?.[0]?.delta ?? {};
          const r = delta.reasoning ?? delta.reasoning_content ?? '';
          const c = delta.content ?? '';
          if ((r || c) && firstTokenAt === null) {
            firstTokenAt = Date.now();
            onFirstToken?.((firstTokenAt - t0) / 1000);
          }
          reasoning += r;
          content += c;

          if (ev.choices?.[0]?.finish_reason) finishReason = ev.choices[0].finish_reason;
          if (ev.usage) usage = ev.usage;
          if (ev.stats) stats = ev.stats;
          if (ev.model_info) modelInfo = ev.model_info;
          if (ev.runtime) runtime = ev.runtime;
        }
      });

      res.on('end', () => {
        const endAt = Date.now();
        const ttft = firstTokenAt ? (firstTokenAt - t0) / 1000 : null;
        const genSeconds = firstTokenAt ? (endAt - firstTokenAt) / 1000 : null;
        const completionTokens = usage?.completion_tokens ?? null;
        resolve({
          choices: [{ message: { content, reasoning_content: reasoning },
                      finish_reason: finishReason }],
          usage: usage ?? {},
          stats: {
            time_to_first_token: stats?.time_to_first_token ?? ttft,
            tokens_per_second: stats?.tokens_per_second ??
              (completionTokens && genSeconds ? completionTokens / genSeconds : null),
            generation_time: stats?.generation_time ?? genSeconds,
            stop_reason: stats?.stop_reason ?? finishReason,
          },
          model_info: modelInfo,
          runtime,
        });
      });
      res.on('error', reject);
    });

    req.on('error', reject);
    req.setTimeout(0);
    req.write(payload);
    req.end();
  });
}

// ---------------------------------------------------------------- scoring
// Port of spark_llm.evals.sec.questions.{parse_number,score_numeric}.

const SCALE = {
  thousand: 1e3, thousands: 1e3, k: 1e3,
  million: 1e6, millions: 1e6, mm: 1e6, mn: 1e6, m: 1e6,
  billion: 1e9, billions: 1e9, bn: 1e9, b: 1e9,
  trillion: 1e12, tn: 1e12,
};
const NUM_RE = /(-|−|\()?\s*(?:US)?\$?\s*(\d[\d,]*(?:\.\d+)?|\.\d+)\s*\)?(?:\s*(thousands?|millions?|billions?|trillion|mm|mn|bn|tn|[kmb])\b)?/gi;

function parseNumber(answer) {
  if (!answer) return null;
  const text = answer.trim().replace(/\*\*/g, '');
  if (/^\W*unknown\W*$/i.test(text || 'x')) return null;
  NUM_RE.lastIndex = 0;
  let m;
  while ((m = NUM_RE.exec(text)) !== null) {
    const raw = m[2].replace(/,/g, '');
    if (raw === '.' || raw === '') continue;
    let value = Number(raw);
    if (Number.isNaN(value)) continue;
    const scale = (m[3] || '').toLowerCase();
    value *= SCALE[scale] ?? 1;
    if (m[1]) value = -value;
    return value;
  }
  return null;
}

function scoreNumeric(answer, expected, alternates = []) {
  const parsed = parseNumber(answer);
  if (parsed === null) return { correct: false, parsed: null, off_by_scale: false, rel_error: null, matched: null };

  const close = (a, b) => {
    if (b === 0) return Math.abs(a) <= 0.5;
    return Math.abs(a - b) <= TOLERANCE * Math.abs(b);
  };
  const rel = expected ? Math.abs(parsed - expected) / Math.abs(expected) : null;
  for (const target of [expected, ...alternates]) {
    if (close(parsed, target)) {
      return { correct: true, parsed, off_by_scale: false, rel_error: rel, matched: target };
    }
  }
  const scaled = [1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9].some((f) => close(parsed * f, expected));
  return { correct: false, parsed, off_by_scale: scaled, rel_error: rel, matched: null };
}

// Same template file as the Python harness (str.format placeholders).
function renderUserPrompt(form, company, periodEnd, document, question, formatHintText) {
  const vars = { form, company, period_end: periodEnd, document, question, format_hint: formatHintText };
  return USER_PROMPT_TEMPLATE.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? String(vars[k]) : m));
}

// ---------------------------------------------------------------- context

function buildContext(text, form, q, budget) {
  const docTokens = estTokens(text);
  if (budget == null || docTokens <= budget) {
    return { text, tokens: docTokens, mode: 'full', fallback: false };
  }
  const sec = sectionText(text, form, FINANCIAL_STATEMENTS);
  if (sec && estTokens(sec) <= budget) {
    return { text: sec, tokens: estTokens(sec), mode: 'section', fallback: true };
  }
  const body = bm25TopChunks(sec ?? text, `${q.text} ${q.label} ${q.tag}`, budget);
  if (!body) {
    return { text: null, tokens: 0, mode: 'chunked', fallback: true,
             reason: `no chunk fits budget ${budget}` };
  }
  return { text: body, tokens: estTokens(body), mode: 'chunked', fallback: true };
}

function budgetFor(nCtx, questionText) {
  if (!nCtx) return null;
  const questionTokens = words(questionText).length * 2;
  return nCtx - MAX_TOKENS - PROMPT_OVERHEAD_TOKENS - questionTokens;
}

// ---------------------------------------------------------------- run

async function cmdRun() {
  const manifestPath = path.join(FILINGS, 'manifest.json');
  if (!existsSync(manifestPath)) throw new Error('No manifest. Run `fetch` first.');
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));

  const models = await lmsModels();
  const loaded = models.find((m) => m.state === 'loaded' && m.type === 'llm');
  const model = process.env.LMS_MODEL || loaded?.id;
  if (!model) throw new Error('No loaded LLM found. Load a model in LM Studio first.');
  const ctx = loaded?.loaded_context_length ?? loaded?.max_context_length ?? 32768;

  let runtimeInfo = await lmsProbe(model);
  const runtimeName = (runtimeInfo?.runtime?.name ?? '').toLowerCase();
  let backendTag = 'lmstudio';
  if (runtimeName.includes('vulkan')) backendTag = 'vulkan';
  else if (runtimeName.includes('rocm') || runtimeName.includes('hip')) backendTag = 'rocm';
  else if (runtimeName.includes('cuda')) backendTag = 'cuda';

  const modelLabel = `${model.split('/').pop()}-halo-${backendTag}`;
  const ts = new Date().toISOString().replace(/[-:]/g, '').replace(/\..+/, 'Z');
  const outDir = path.join(EVALS, modelLabel, `${ts}-extract-full`);
  await mkdir(outDir, { recursive: true });
  const resultsPath = path.join(outDir, 'results.jsonl');

  const onlyTicker = flag('ticker');
  const onlyTag = flag('tag');
  const onlyForms = flag('forms') ? String(flag('forms')).split(',').map((s) => s.trim()) : null;
  const limit = flag('limit') ? Number(flag('limit')) : Infinity;

  console.log(`model   ${model}`);
  console.log(`context ${ctx.toLocaleString()} tok`);
  console.log(`out     ${outDir}`);
  if (REASONING_EFFORT && REASONING_EFFORT !== true) {
    console.log(`effort  ${REASONING_EFFORT}`);
  }
  console.log();

  if (runtimeInfo?.runtime) {
    console.log(`backend ${runtimeInfo.runtime.name} v${runtimeInfo.runtime.version}\n`);
  }

  let n = 0, correct = 0, skipped = 0, truncated = 0, fallbacks = 0;
  let offByScale = 0;
  const wallStart = Date.now();
  const byMode = {}, byForm = {};
  const ttfts = [], totals = [], promptTps = [], decodeTps = [];
  let promptTokens = 0, reasoningChars = 0;
  const seenFilings = new Set();

  outer:
  for (const f of manifest.filings) {
    if (onlyTicker && f.ticker !== onlyTicker) continue;
    if (onlyForms && !onlyForms.includes(f.form)) continue;
    const text = await readFile(f.text_path, 'utf8');
    const docTokens = estTokens(text);

    for (const q of f.questions) {
      if (onlyTag && q.tag !== onlyTag) continue;
      if (n >= limit) break outer;

      const warm = seenFilings.has(f.key);
      seenFilings.add(f.key);

      const budget = budgetFor(ctx, q.text);
      const ctxBuilt = buildContext(text, f.form, q, budget);
      if (ctxBuilt.fallback) fallbacks++;

      const base = {
        id: q.id,
        ticker: f.ticker,
        form: f.form,
        report_date: f.report_date,
        tag: q.tag,
        kind: q.kind,
        unit: q.unit,
        expected: q.expected,
        concept: q.concept,
        alternates: q.alternates,
        question: q.text,
        warm,
        mode: ctxBuilt.mode,
        fallback: ctxBuilt.fallback,
        doc_tokens: docTokens,
        context_tokens: ctxBuilt.tokens,
      };

      if (ctxBuilt.text == null) {
        skipped++;
        await appendFile(resultsPath, JSON.stringify({
          ...base, skipped: true, reason: ctxBuilt.reason, correct: null,
        }) + '\n');
        console.log(`  ${q.id.padEnd(58)} skip  ${ctxBuilt.reason}`);
        continue;
      }

      const messages = [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: renderUserPrompt(
            f.form, f.company, f.report_date, ctxBuilt.text, q.text, q.format_hint) },
      ];

      const t0 = Date.now();
      let resp;
      try {
        resp = await lmsStream(model, messages, (ttft) => {
          console.log(`  ${q.id.padEnd(58)} first token after ${ttft.toFixed(1)}s`);
        });
      } catch (err) {
        const cause = err.cause ? ` (${err.cause.code ?? err.cause.message ?? err.cause})` : '';
        console.log(`  ${q.id.padEnd(58)} ERROR ${err.message}${cause}`);
        skipped++;
        await appendFile(resultsPath, JSON.stringify({
          ...base, skipped: true, error: String(err),
          cause: err.cause ? String(err.cause.code ?? err.cause.message) : null,
          correct: null,
        }) + '\n');
        continue;
      }
      const wall = (Date.now() - t0) / 1000;

      if (!runtimeInfo?.runtime && resp.runtime) {
        runtimeInfo = { runtime: resp.runtime, model_info: resp.model_info };
      }
      const choice = resp.choices?.[0];
      const answer = (choice?.message?.content ?? '').trim();
      const reasoning = choice?.message?.reasoning_content ?? '';
      reasoningChars += reasoning.length;
      const stats = resp.stats ?? {};
      if (stats.time_to_first_token != null) ttfts.push(stats.time_to_first_token);
      totals.push(wall);
      if (stats.tokens_per_second != null) decodeTps.push(stats.tokens_per_second);
      const pTok = resp.usage?.prompt_tokens
        || estTokens(messages.map((m) => m.content).join('\n'));
      promptTokens += pTok;
      const prefill = stats.time_to_first_token;
      if (prefill > 0 && pTok) promptTps.push(pTok / prefill);

      const isTruncated = choice?.finish_reason === 'length'; // same rule as ChatResult.truncated
      if (isTruncated) truncated++;

      const s = scoreNumeric(answer, q.expected, q.alternates);
      n++;
      if (s.correct) correct++;
      if (s.off_by_scale) offByScale++;
      byMode[ctxBuilt.mode] ??= { n: 0, correct: 0 };
      byMode[ctxBuilt.mode].n++;
      if (s.correct) byMode[ctxBuilt.mode].correct++;
      byForm[f.form] ??= { n: 0, correct: 0 };
      byForm[f.form].n++;
      if (s.correct) byForm[f.form].correct++;

      await appendFile(resultsPath, JSON.stringify({
        ...base,
        answer: answer.slice(0, 300),
        parsed: s.parsed,
        correct: s.correct,
        matched_value: s.matched,
        off_by_scale: s.off_by_scale,
        rel_error: s.rel_error,
        truncated: isTruncated,
        skipped: false,
        ttft_s: stats.time_to_first_token ?? null,
        total_s: wall,
        prompt_tokens: pTok,
        completion_tokens: resp.usage?.completion_tokens ?? 0,
        reasoning_chars: reasoning.length,
        prompt_tps: (prefill > 0 && pTok) ? pTok / prefill : null,
        decode_tps: stats.tokens_per_second ?? null,
      }) + '\n');

      const mark = s.correct ? 'ok  ' : (isTruncated ? 'trunc' : (s.off_by_scale ? 'scale' : 'miss'));
      console.log(`  ${q.id.padEnd(58)} ${mark} ${ctxBuilt.mode.padEnd(8)} ` +
                  `${String(pTok).padStart(7)} tok  ${wall.toFixed(1)}s`);
    }
  }

  const decoding = { temperature: 0, seed: SEED, max_tokens: MAX_TOKENS };
  if (REASONING_EFFORT && REASONING_EFFORT !== true) {
    decoding.reasoning_effort = REASONING_EFFORT;
  }

  const configHash = hash({
    scoring_version: CONFIG.scoring_version ?? null,
    tags: TAGS,                       // full definitions: alias/label changes must change the hash
    tickers: COMPANIES.map((c) => c.ticker),
    forms: FORMS,
    ...decoding,
    tol: TOLERANCE,
    chunk_tokens: CHUNK_TOKENS,
    top_k: TOP_K,
  });
  const startedIso = new Date(wallStart).toISOString().replace(/\.\d{3}Z$/, '+00:00');
  const finishedIso = new Date().toISOString().replace(/\.\d{3}Z$/, '+00:00');
  const rtName = runtimeInfo?.runtime?.name ?? 'unknown';
  const rtVersion = runtimeInfo?.runtime?.version ?? null;
  // Same shape as spark_llm.evals.runs.RunRecord so `local-llm eval report` reads it directly.
  const run = {
    suite: 'sec',
    task: 'extract-full',
    model: model.split('/').pop(),  // bare name; the folder label carries platform/backend
    started: startedIso,
    finished: finishedIso,
    provenance: {
      timestamp: startedIso,
      hostname: os.hostname(),
      machine: process.arch,
      tool_version: `lmstudio.mjs/${rtVersion ?? '?'}`,
      llama_cpp_pinned: null,
      llama_cpp_checkout: null,
      build_id: `${backendTag}:lmstudio-${rtVersion ?? '?'}`,
      binary_version: rtName,
      gpu: {},
      platform: 'halo',
      backend: backendTag,
      llama_cpp_release: rtVersion,
    },
    server: {
      n_ctx_per_slot: ctx,
      served_via: 'lmstudio',
      runtime: rtName,
      runtime_version: rtVersion,
    },
    runtime: {
      served_via: 'lmstudio',
      model_id: model,
      quant: runtimeInfo?.model_info?.quant ?? null,
      backend_raw: rtName,
    },
    quality: decoding,
    task_config: {
      scoring_version: CONFIG.scoring_version ?? null,
      no_document: false,
      tolerance: TOLERANCE,
      chunk_tokens: CHUNK_TOKENS,
      top_k: TOP_K,
      token_estimate: `chars/${CHARS_PER_TOKEN} (approximate)`,
      forms: onlyForms,
    },
    config_hash: configHash,
    tools: {},
    summary: {
      n, correct, skipped, truncated, fallback: fallbacks, off_by_scale: offByScale,
      score: n ? correct / n : 0,
      by_mode: byMode, by_form: byForm,
      ttft_p50_s: p50(ttfts), total_p50_s: p50(totals), total_p95_s: p95(totals),
      prompt_tps_p50: p50(promptTps), decode_tps_p50: p50(decodeTps),
      total_tokens: promptTokens, cached_prompt_tokens: null, reasoning_chars: reasoningChars,
      wall_clock_s: (Date.now() - wallStart) / 1000,
    },
    n_results: n + skipped,
    format: 'runrecord/1',
    model_label: modelLabel,
  };
  await writeFile(path.join(outDir, 'run.json'), JSON.stringify(run, null, 2));

  console.log(`\n${correct}/${n} = ${pct(correct, n)}  (fallback ${fallbacks}, truncated ${truncated}, skipped ${skipped})`);
  console.log(`run.json -> ${outDir}`);
}

// ---------------------------------------------------------------- report

async function cmdReport() {
  if (!existsSync(EVALS)) throw new Error('No runs yet.');
  const runs = [];
  for (const modelDir of await readdir(EVALS)) {
    const modelPath = path.join(EVALS, modelDir);
    if (!(await stat(modelPath)).isDirectory()) continue;
    // Skip non-run trees if someone points EVALS at state/evals instead of …/sec.
    if (modelDir === 'sec') continue;
    const dirs = (await readdir(modelPath)).sort();
    for (const d of dirs) {
      const runPath = path.join(modelPath, d);
      if (!(await stat(runPath)).isDirectory()) continue;
      const rp = path.join(runPath, 'run.json');
      if (!existsSync(rp)) continue;
      const run = JSON.parse(await readFile(rp, 'utf8'));
      const lines = (await readFile(path.join(runPath, 'results.jsonl'), 'utf8'))
        .trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));
      runs.push({ run, results: lines, dir: d });
    }
  }
  // Accessors: RunRecord shape (summary/provenance/server nested) or the old flat layout.
  const S = (run) => run.summary ?? run;
  const P = (run) => run.provenance ?? run;
  const L = (run) => run.model_label ?? run.model;
  const ctxOf = (run) => run.server?.n_ctx_per_slot ?? run.context;
  const latest = new Map();
  for (const r of runs) latest.set(L(r.run), r);
  const sel = [...latest.values()].sort((a, b) => (S(b.run).score ?? 0) - (S(a.run).score ?? 0));
  if (!sel.length) { console.log('no runs'); return; }

  const row = (c) => c.join('\t');
  console.log(row(['model', 'task', 'n', 'score', 'skipped', 'truncated', 'scale',
                   'ttft p50 s', 'total p50 s', 'total p95 s', 'prompt t/s', 'decode t/s',
                   'tokens', 'cached', 'reasoning', 'ctx', 'build', 'platform', 'note']));
  for (const { run } of sel) {
    const s = S(run), p = P(run);
    console.log(row([
      L(run), run.task, s.n, (s.score ?? 0).toFixed(3), s.skipped, s.truncated,
      s.off_by_scale ?? '-',
      (s.ttft_p50_s ?? s.ttft_p50)?.toFixed(3) ?? '-', (s.total_p50_s ?? s.total_p50)?.toFixed(2) ?? '-',
      (s.total_p95_s ?? s.total_p95)?.toFixed(2) ?? '-',
      s.prompt_tps_p50?.toFixed(0) ?? '-', s.decode_tps_p50?.toFixed(0) ?? '-',
      s.total_tokens ?? s.prompt_tokens, s.cached_prompt_tokens ?? 'n/a', s.reasoning_chars,
      ctxOf(run),
      `${p.backend ?? 'api'}/${run.server?.runtime_version ?? run.runtime_version ?? p.llama_cpp_release ?? p.llama_cpp_checkout ?? '?'}`,
      `${p.platform ?? (run.server?.provider ? 'api' : 'spark')}/${run.server?.served_via ?? run.served_via ?? (run.server?.provider ?? 'llama-server')}`,
      s.fallback ? `fallback ${s.fallback}` : '-',
    ]));
  }

  const hashes = new Set(sel.map((s) => s.run.config_hash));
  if (hashes.size > 1) console.log(`warning: runs use different configs (${[...hashes].join(', ')})`);

  const idSets = sel.map((s) => new Set(s.results.filter((r) => !r.error && !r.skipped).map((r) => r.id)));
  const common = idSets.length ? [...idSets[0]].filter((id) => idSets.every((s) => s.has(id))) : [];
  console.log('\nPaired (items answered by every run of the task)');
  console.log(row(['task', 'model', 'paired n', 'correct', 'paired score']));
  for (const { run, results } of sel) {
    const hits = results.filter((r) => common.includes(r.id) && r.correct).length;
    console.log(row([run.task, L(run), common.length, hits, pct(hits, common.length)]));
  }

  console.log('\nContext mode');
  console.log(row(['model', 'fallback', 'by_mode']));
  for (const { run } of sel) {
    const modes = Object.entries(S(run).by_mode ?? {})
      .map(([m, v]) => `${m} ${v.correct}/${v.n}`).join(', ');
    console.log(row([L(run), S(run).fallback, modes]));
  }

  console.log('\nBy form');
  console.log(row(['form', 'n', ...sel.map((s) => L(s.run))]));
  const forms = [...new Set(sel.flatMap((s) => Object.keys(S(s.run).by_form ?? {})))].sort();
  for (const form of forms) {
    const cells = sel.map((s) => {
      const v = S(s.run).by_form?.[form];
      return v ? `${v.correct}/${v.n} (${pct(v.correct, v.n)})` : '-';
    });
    const nk = S(sel[0].run).by_form?.[form]?.n ?? 0;
    console.log(row([form, nk, ...cells]));
  }

  for (const [label, key] of [['By tag', 'tag'], ['By company', 'ticker']]) {
    console.log(`\n${label}`);
    console.log(row([key === 'tag' ? 'tag' : 'ticker', 'n', ...sel.map((s) => L(s.run))]));
    const keys = [...new Set(sel.flatMap((s) => s.results.map((r) => r[key])))].filter(Boolean).sort();
    for (const k of keys) {
      const cells = sel.map((s) => {
        const rs = s.results.filter((r) => r[key] === k && !r.error && !r.skipped);
        const c = rs.filter((r) => r.correct).length;
        return `${c}/${rs.length} (${pct(c, rs.length)})`;
      });
      const nk = sel[0].results.filter((r) => r[key] === k && !r.error && !r.skipped).length;
      console.log(row([k, nk, ...cells]));
    }
  }

  console.log('\nDisagreements');
  console.log(row(['id', 'ticker', ...sel.map((s) => L(s.run))]));
  for (const id of common) {
    const cells = sel.map((s) => {
      const r = s.results.find((x) => x.id === id);
      const base = r.correct ? 'ok' : (r.off_by_scale ? 'scale' : 'miss');
      return r.mode === 'full' ? base : `${base}/${r.mode}`;
    });
    if (new Set(cells.map((c) => c.split('/')[0])).size > 1) {
      console.log(row([id, id.split(':')[0], ...cells]));
    }
  }
}

// ---------------------------------------------------------------- main

const commands = { fetch: cmdFetch, run: cmdRun, report: cmdReport };
if (!commands[CMD]) {
  console.log('usage: node scripts/lmstudio.mjs <fetch|run|report> [--limit N] [--ticker X] [--tag Y] [--forms 10-K] [--reasoning-effort low] [--insecure]');
  process.exit(1);
}
commands[CMD]().catch((err) => { console.error(`\n${err.message}`); process.exit(1); });
