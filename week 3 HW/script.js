/**
 * script.js — Matrix Factorization Recommender (vanilla JS, in-browser SGD).
 * Biased MF: r̂_ui = μ + b_u + b_i + p_u^T q_i
 */
'use strict';

// ---------- Tiny helpers ----------
const $ = (id) => document.getElementById(id);
const setStatus = (s) => { const el = $('status'); if (el) el.textContent = s; };
const setBar = (p) => { const el = $('bar'); if (el) el.style.width = `${Math.max(0,Math.min(100,p))}%`; };
const logln = (s) => { const el = $('log'); if (el) { el.textContent += s + '\\n'; el.scrollTop = el.scrollHeight; } };
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

// Deterministic PRNG (xorshift32) for reproducible shuffles/splits
function XorShift32(seed) {
  let x = seed >>> 0;
  return function() {
    x ^= x << 13; x >>>= 0;
    x ^= x >> 17; x >>>= 0;
    x ^= x << 5;  x >>>= 0;
    return (x >>> 0) / 4294967296; // [0,1)
  };
}

// ---------- Global state ----------
let userIds = [];              // distinct user IDs (as in data)
let itemIds = [];              // distinct item IDs (as in data)
let uid2idx = new Map();       // userId -> 0..U-1
let iid2idx = new Map();       // itemId -> 0..I-1

let K = 32;
let mu = 0;
let bu = null, bi = null;
let P = null, Q = null;

let trained = false;

// ---------- Build indices & populate UI ----------
function buildIndicesAndUI() {
  const uSet = new Set(), iSet = new Set();
  for (const r of ratings) { uSet.add(r.userId); iSet.add(r.itemId); }
  userIds = Array.from(uSet).sort((a,b)=>a-b);
  itemIds = Array.from(iSet).sort((a,b)=>a-b);
  uid2idx = new Map(userIds.map((id,idx)=>[id, idx]));
  iid2idx = new Map(itemIds.map((id,idx)=>[id, idx]));

  const sel = $('user-select');
  sel.innerHTML = '<option value="" selected disabled>Select a user…</option>';
  for (const uid of userIds) {
    const opt = document.createElement('option');
    opt.value = String(uid);
    opt.textContent = `User ${uid}`;
    sel.appendChild(opt);
  }
  $('train-btn').disabled = ratings.length === 0;
  $('rec-btn').disabled = true;
  setStatus(`Data ready. Users=${userIds.length}, Items=${itemIds.length}, Ratings=${ratings.length}`);
}

// ---------- Model init/pred ----------
function initModel(K_) {
  const U = userIds.length, I = itemIds.length;
  K = K_;
  mu = ratings.reduce((s,r)=>s+r.rating, 0) / Math.max(1, ratings.length);
  bu = new Float64Array(U);
  bi = new Float64Array(I);
  P = new Float64Array(U * K);
  Q = new Float64Array(I * K);
  const scale = 0.1 / Math.sqrt(K);
  for (let u = 0; u < U; u++) for (let f = 0; f < K; f++) P[u*K + f] = (Math.random() - 0.5) * 2 * scale;
  for (let i = 0; i < I; i++) for (let f = 0; f < K; f++) Q[i*K + f] = (Math.random() - 0.5) * 2 * scale;
  trained = false;
}

function predictIdx(uIdx, iIdx) {
  let s = mu + bu[uIdx] + bi[iIdx];
  const pu = uIdx * K, qi = iIdx * K;
  for (let f = 0; f < K; f++) s += P[pu + f] * Q[qi + f];
  return s;
}

// ---------- Train/val split & RMSE ----------
function trainValSplit(valFrac, seed) {
  const rnd = XorShift32(seed || 1337);
  const idxs = ratings.map((_, idx) => idx);
  for (let i = idxs.length - 1; i > 0; i--) {
    const j = (rnd() * (i + 1)) | 0;
    const tmp = idxs[i]; idxs[i] = idxs[j]; idxs[j] = tmp;
  }
  const nVal = Math.floor(idxs.length * valFrac);
  const train = [], val = [];
  for (let k = 0; k < idxs.length; k++) {
    const r = ratings[idxs[k]];
    (k < nVal ? val : train).push(r);
  }
  return { train, val };
}

function rmseOf(set) {
  let se = 0, n = 0;
  for (const r of set) {
    const u = uid2idx.get(r.userId);
    const i = iid2idx.get(r.itemId);
    if (u === undefined || i === undefined) continue;
    const e = r.rating - predictIdx(u, i);
    se += e * e; n += 1;
  }
  return n ? Math.sqrt(se / n) : NaN;
}

// ---------- Training ----------
async function trainModel() {
  const K_      = clamp(parseInt($('factors').value, 10) || 32, 4, 128);
  const epochs  = clamp(parseInt($('epochs').value, 10) || 10, 1, 100);
  const lr0     = parseFloat($('lr').value) || 0.01;
  const reg     = parseFloat($('reg').value) || 0.05;
  const valFrac = clamp(parseFloat($('val-split').value) || 0.1, 0, 0.9);
  const seed    = clamp(parseInt($('seed').value, 10) || 1337, 1, 2147483647);

  if (ratings.length === 0) { setStatus('No ratings loaded. Load data first.'); return; }
  initModel(K_);
  const { train, val } = trainValSplit(valFrac, seed);
  setStatus(`Training (train=${train.length}, val=${val.length}) ...`);
  $('train-btn').disabled = true;
  $('rec-btn').disabled = true;
  setBar(0);

  const idxs = train.map((_, idx) => idx);
  const rnd = XorShift32(seed ^ 0x9e3779b9);

  const t0 = performance.now();
  for (let ep = 1; ep <= epochs; ep++) {
    for (let i = idxs.length - 1; i > 0; i--) {
      const j = (rnd() * (i + 1)) | 0;
      const tmp = idxs[i]; idxs[i] = idxs[j]; idxs[j] = tmp;
    }

    let se = 0, count = 0;
    const lr = lr0;
    for (const k of idxs) {
      const r = train[k];
      const u = uid2idx.get(r.userId);
      const i = iid2idx.get(r.itemId);
      if (u === undefined || i === undefined) continue;

      const pu = u * K, qi = i * K;
      let pred = mu + bu[u] + bi[i];
      for (let f = 0; f < K; f++) pred += P[pu + f] * Q[qi + f];
      const e = r.rating - pred;
      se += e * e;

      bu[u] += lr * (e - reg * bu[u]);
      bi[i] += lr * (e - reg * bi[i]);
      for (let f = 0; f < K; f++) {
        const puf = P[pu + f];
        const qif = Q[qi + f];
        P[pu + f] += lr * (e * qif - reg * puf);
        Q[qi + f] += lr * (e * puf - reg * qif);
      }

      if ((++count % 4000) === 0) {
        setBar((count / idxs.length) * 100);
        await new Promise(requestAnimationFrame);
      }
    }

    const trainRmse = Math.sqrt(se / Math.max(1, idxs.length));
    const valRmse = rmseOf(val);
    logln(`Epoch ${ep}/${epochs} — train RMSE: ${isNaN(trainRmse)?'n/a':trainRmse.toFixed(4)} | val RMSE: ${isNaN(valRmse)?'n/a':valRmse.toFixed(4)}`);
    setStatus(`Epoch ${ep}/${epochs} done.`);
    setBar(100);
    await new Promise(requestAnimationFrame);
    setBar(0);
  }
  const t1 = performance.now();
  trained = true;
  $('train-btn').disabled = false;
  $('rec-btn').disabled = false;
  setStatus(`Training finished in ${(t1 - t0).toFixed(0)} ms. Model is ready.`);
  $('result').textContent = 'Pick a user and click "Get Recommendations".';
}

// ---------- Recommendation ----------
function getRecommendations() {
  const resultEl = $('result');
  if (!trained) { resultEl.textContent = 'Train the model first.'; return; }

  const userSel = $('user-select');
  const userVal = userSel ? parseInt(userSel.value, 10) : NaN;
  if (!userVal) { resultEl.textContent = 'Please select a user.'; return; }
  const uIdx = uid2idx.get(userVal);
  if (uIdx === undefined) { resultEl.textContent = 'Unknown user.'; return; }

  const topN = clamp(parseInt($('top-n').value, 10) || 10, 1, 50);
  const excludeSeen = $('exclude-seen')?.checked ?? true;

  const seen = new Set();
  if (excludeSeen) for (const r of ratings) if (r.userId === userVal) seen.add(r.itemId);

  const scored = [];
  for (const itemId of itemIds) {
    if (excludeSeen && seen.has(itemId)) continue;
    const iIdx = iid2idx.get(itemId);
    const pred = predictIdx(uIdx, iIdx);
    const clipped = clamp(pred, 0.5, 5.0);
    scored.push({ itemId, pred: clipped });
  }
  scored.sort((a,b)=> b.pred - a.pred);

  const top = scored.slice(0, topN).map(s => {
    const m = movies.find(mm => mm.id === s.itemId);
    return `${m ? m.title : `Movie ${s.itemId}`} — predicted ${s.pred.toFixed(2)}/5`;
  });

  resultEl.textContent = top.length
    ? `Top ${topN} for User ${userVal}: ` + top.join('; ')
    : `No items to recommend for User ${userVal}.`;
}

// ---------- Boot ----------
window.addEventListener('load', async () => {
  try {
    await loadData();                 // try fetch
    if (ratings.length > 0) buildIndicesAndUI();
    else setStatus('No data fetched. You can upload files above and click "Use uploaded files".');
  } catch (e) {
    console.error(e);
  }

  $('use-uploads')?.addEventListener('click', async () => {
    const fItem = $('file-item').files[0];
    const fData = $('file-data').files[0];
    try {
      await loadDataFromFiles(fItem, fData);
      buildIndicesAndUI();
    } catch (_) {}
  });

  $('train-btn')?.addEventListener('click', () => { trainModel(); });
  $('rec-btn')?.addEventListener('click', () => { getRecommendations(); });
});
