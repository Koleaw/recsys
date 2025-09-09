/**
 * script.js — UI & recommendations (v7).
 */
'use strict';
console.log('[script.js v7] loaded');

/** Jaccard similarity: |A ∩ B| / |A ∪ B| */
function jaccard(setA, setB) {
  if (setA.size === 0 && setB.size === 0) return 0;
  let inter = 0;
  for (const v of setA) if (setB.has(v)) inter++;
  const union = new Set([...setA, ...setB]).size;
  return union === 0 ? 0 : inter / union;
}

/** 0..1 → "NN%" */
function pct(x, digits = 0) {
  return `${(x * 100).toFixed(digits)}%`;
}

/** Init after window load (скрипты подключены внизу, без defer) */
window.addEventListener('load', async () => {
  const resultEl = document.getElementById('result');
  try {
    await loadData(); // from data.js
    populateMoviesDropdown();
    if (resultEl) {
      resultEl.innerText = movies.length
        ? 'Data loaded. Please select a movie.'
        : 'Data loaded, but no movies were parsed. Check u.item.';
    }
  } catch (e) {
    // loadData уже написал ошибку в #result; просто зафиксируем в консоли
    console.error('[script.js] init failed:', e);
  }
});

/** Fill dropdown alphabetically */
function populateMoviesDropdown() {
  const select = document.getElementById('movie-select');
  if (!select) return;

  select.innerHTML = '<option value="" disabled selected>Select a movie…</option>';
  const sorted = [...movies].sort((a, b) => a.title.localeCompare(b.title));

  for (const m of sorted) {
    const opt = document.createElement('option');
    opt.value = String(m.id);
    opt.textContent = m.title;
    select.appendChild(opt);
  }
}

/** Main: compute Jaccard and show top-2 with % */
function getRecommendations() {
  const select = document.getElementById('movie-select');
  const resultEl = document.getElementById('result');

  const selectedVal = select ? select.value : '';
  if (!selectedVal) { if (resultEl) resultEl.innerText = 'Please select a movie first.'; return; }
  const selectedId = parseInt(selectedVal, 10);

  const likedMovie = movies.find(m => m.id === selectedId);
  if (!likedMovie) { if (resultEl) resultEl.innerText = 'Selected movie not found.'; return; }

  const likedSet = new Set(likedMovie.genres);
  const candidates = movies.filter(m => m.id !== likedMovie.id);

  const scored = candidates.map(m => ({
    ...m,
    score: jaccard(likedSet, new Set(m.genres))
  })).sort((a, b) => (b.score !== a.score ? b.score - a.score : a.title.localeCompare(b.title)));

  const top = scored.slice(0, 2);
  if (!top.length) { if (resultEl) resultEl.innerText = `No recommendations available for "${likedMovie.title}".`; return; }

  const list = top.map(m => `${m.title} (${pct(m.score, 0)})`).join(', ');
  if (resultEl) {
    resultEl.innerText = likedSet.size === 0
      ? `Heads up: "${likedMovie.title}" has no genre tags in your data, so similarities are 0%. Recommended: ${list}.`
      : `Because you liked "${likedMovie.title}", we recommend: ${list}.`;
  }
}
window.getRecommendations = getRecommendations;
