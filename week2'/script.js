/**
 * script.js – UI & recommendation logic module.
 * Depends on data.js having loaded `movies`, `ratings`, and `loadData()`.
 */

// --- Helpers ---
/** Jaccard similarity between two Sets. J(A,B) = |A ∩ B| / |A ∪ B| */
function jaccard(setA, setB) {
  if (setA.size === 0 && setB.size === 0) return 0;
  let intersection = 0;
  for (const v of setA) if (setB.has(v)) intersection++;
  const unionSize = new Set([...setA, ...setB]).size;
  return unionSize === 0 ? 0 : intersection / unionSize;
}

/** Format 0..1 to percent string */
function pct(x, digits = 0) {
  return `${(x * 100).toFixed(digits)}%`;
}

// Initialize app once the page is ready
window.onload = async () => {
  const resultEl = document.getElementById('result');
  if (resultEl) resultEl.innerText = 'Loading data...';

  await loadData();

  populateMoviesDropdown();

  if (resultEl) {
    if (movies.length === 0) {
      resultEl.innerText = 'Data loaded, but no movies were parsed. Check u.item format/path.';
    } else {
      resultEl.innerText = 'Data loaded. Please select a movie.';
    }
  }
};

/**
 * Populate the #movie-select dropdown with alphabetized movie titles.
 */
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

/**
 * Main handler called by the button click.
 * Follows the 7 steps outlined in the specification, and displays similarity in %.
 */
function getRecommendations() {
  const select = document.getElementById('movie-select');
  const resultEl = document.getElementById('result');

  // Step 1: Get user input
  const selectedVal = select ? select.value : '';
  if (!selectedVal) {
    if (resultEl) resultEl.innerText = 'Please select a movie first.';
    return;
  }
  const selectedId = parseInt(selectedVal, 10);

  // Step 2: Find liked movie
  const likedMovie = movies.find(m => m.id === selectedId);
  if (!likedMovie) {
    if (resultEl) resultEl.innerText = 'Selected movie not found. Please try another.';
    return;
  }

  // Step 3: Prepare sets and candidate list
  const likedGenresSet = new Set(likedMovie.genres);
  const candidateMovies = movies.filter(m => m.id !== likedMovie.id);

  // Step 4: Calculate Jaccard scores
  const scoredMovies = candidateMovies.map(m => {
    const candSet = new Set(m.genres);
    const score = jaccard(likedGenresSet, candSet);
    return { ...m, score };
  });

  // Step 5: Sort by score (desc), tie-break by title
  scoredMovies.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return a.title.localeCompare(b.title);
  });

  // Step 6: Take top 2
  const top = scoredMovies.slice(0, 2);

  // Step 7: Display result with % similarity
  if (top.length === 0) {
    if (resultEl) resultEl.innerText = `No recommendations available for "${likedMovie.title}".`;
    return;
  }

  const recList = top.map(m => `${m.title} (${pct(m.score, 0)})`).join(', ');

  if (resultEl) {
    if (likedGenresSet.size === 0) {
      resultEl.innerText =
        `Heads up: "${likedMovie.title}" has no genre tags in your data, so all similarities are 0%. ` +
        `Because you liked "${likedMovie.title}", we recommend: ${recList}.`;
    } else {
      resultEl.innerText =
        `Because you liked "${likedMovie.title}", we recommend: ${recList}.`;
    }
  }
}

// Expose the handler globally so the inline onclick works
window.getRecommendations = getRecommendations;
