/**
 * script.js – UI & recommendation logic module.
 * Depends on data.js having loaded `movies`, `ratings`, and `loadData()`.
 */

// Initialize app once the page is ready
window.onload = async () => {
  const resultEl = document.getElementById('result');
  if (resultEl) resultEl.innerText = 'Loading data...';

  await loadData();

  populateMoviesDropdown();

  if (resultEl) {
    resultEl.innerText = 'Data loaded. Please select a movie.';
  }
};

/**
 * Populate the #movie-select dropdown with alphabetized movie titles.
 */
function populateMoviesDropdown() {
  const select = document.getElementById('movie-select');
  if (!select) return;

  // Reset with placeholder
  select.innerHTML = '<option value="" disabled selected>Select a movie…</option>';

  // Sort for UX friendliness
  const sorted = [...movies].sort((a, b) => a.title.localeCompare(b.title));

  for (const m of sorted) {
    const opt = document.createElement('option');
    opt.value = String(m.id);
    opt.textContent = m.title;
    select.appendChild(opt);
  }
}

/**
 * Compute Jaccard similarity between two Sets.
 * J(A,B) = |A ∩ B| / |A ∪ B|
 * @param {Set<string>} setA
 * @param {Set<string>} setB
 * @returns {number}
 */
function jaccard(setA, setB) {
  if (setA.size === 0 && setB.size === 0) return 0;

  let intersection = 0;
  for (const v of setA) {
    if (setB.has(v)) intersection++;
  }

  const unionSize = new Set([...setA, ...setB]).size;
  return unionSize === 0 ? 0 : intersection / unionSize;
}

/**
 * Main handler called by the button click.
 * Follows the 7 steps outlined in the specification.
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

  // Step 6: Take top N (2 as per spec)
  const top = scoredMovies.slice(0, 2);

  // Step 7: Display result
  if (top.length === 0) {
    if (resultEl) {
      resultEl.innerText = `No recommendations available for "${likedMovie.title}".`;
    }
    return;
  }

  const recList = top.map(m => m.title).join(', ');
  if (resultEl) {
    resultEl.innerText = `Because you liked "${likedMovie.title}", we recommend: ${recList}.`;
  }
}

// Expose the handler globally so the inline onclick works
window.getRecommendations = getRecommendations;
