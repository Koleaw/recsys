/**
 * script.js — UI & recommendations (Cosine Similarity), v10.
 * We build 18-dim binary genre vectors and compute cosine similarity.
 */
'use strict';
console.log('[script.js v10] loaded');

/** The 18 named genres (same order as in data.js) */
const GENRE_LIST = [
  "Action","Adventure","Animation","Children's","Comedy","Crime","Documentary",
  "Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance",
  "Sci-Fi","Thriller","War","Western"
];

/** Convert array of genre names -> fixed-length binary vector (Float32Array length 18) */
function genresToVector(genreArr) {
  const v = new Float32Array(GENRE_LIST.length);
  const set = new Set(genreArr);
  for (let i = 0; i < GENRE_LIST.length; i++) {
    if (set.has(GENRE_LIST[i])) v[i] = 1;
  }
  return v;
}

/** Cosine similarity between two numeric vectors of equal length */
function cosineSim(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    const ai = a[i], bi = b[i];
    dot += ai * bi;
    na += ai * ai;
    nb += bi * bi;
  }
  if (na === 0 || nb === 0) return 0;         // one (or both) is a zero vector
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

/** Format 0..1 as "NN%" */
function pct(x, digits = 0) {
  return `${(x * 100).toFixed(digits)}%`;
}

/** Init */
window.addEventListener('load', async () => {
  const resultEl = document.getElementById('result');
  try {
    if (resultEl) resultEl.innerText = 'Loading data...';
    await loadData(); // from data.js
    populateMoviesDropdown();
    if (resultEl) {
      resultEl.innerText = movies.length
        ? 'Data loaded. Please select a movie.'
        : 'Data loaded, but no movies were parsed. Check u.item.';
    }
  } catch (e) {
    console.error('[script.js] init failed:', e);
    if (resultEl) resultEl.innerText = `Init error: ${e.message}`;
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

/** Main: compute Cosine Similarity and show top-2 with labelled percentages */
function getRecommendations() {
  const select = document.getElementById('movie-select');
  const resultEl = document.getElementById('result');

  const selectedVal = select ? select.value : '';
  if (!selectedVal) { if (resultEl) resultEl.innerText = 'Please select a movie first.'; return; }
  const selectedId = parseInt(selectedVal, 10);

  const likedMovie = movies.find(m => m.id === selectedId);
  if (!likedMovie) { if (resultEl) resultEl.innerText = 'Selected movie not found.'; return; }

  // Build vectors
  const likedVec = genresToVector(likedMovie.genres);

  // Score all candidates
  const scored = movies
    .filter(m => m.id !== likedMovie.id)
    .map(m => {
      const v = genresToVector(m.genres);
      return { ...m, score: cosineSim(likedVec, v) };
    })
    .sort((a, b) => (b.score !== a.score ? b.score - a.score : a.title.localeCompare(b.title)));

  const top = scored.slice(0, 2);
  if (!top.length) { if (resultEl) resultEl.innerText = `No recommendations available for "${likedMovie.title}".`; return; }

  // Show "Cosine XX%" near each title
  const list = top.map(m => `${m.title} (Cosine ${pct(m.score, 0)})`).join(', ');
  if (resultEl) {
    resultEl.innerText = `Because you liked "${likedMovie.title}", we recommend: ${list}.`;
  }
}

window.getRecommendations = getRecommendations;
