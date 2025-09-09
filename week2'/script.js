/**
 * script.js — UI + рекомендации. Версия: v5 (печатаем в консоль для проверки кэша).
 */
console.log('script.js v5 loaded');

/** Индекс Жаккара по жанрам */
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

window.onload = async () => {
  const resultEl = document.getElementById('result');
  if (resultEl) resultEl.innerText = 'Loading data...';
  await loadData();
  populateMoviesDropdown();
  if (resultEl) {
    resultEl.innerText = movies.length
      ? 'Данные загружены. Выберите фильм.'
      : 'Данные загружены, но фильмы не распознаны (проверьте u.item).';
  }
};

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

function getRecommendations() {
  const select = document.getElementById('movie-select');
  const resultEl = document.getElementById('result');

  const selectedVal = select ? select.value : '';
  if (!selectedVal) { if (resultEl) resultEl.innerText = 'Сначала выберите фильм.'; return; }
  const selectedId = parseInt(selectedVal, 10);

  const likedMovie = movies.find(m => m.id === selectedId);
  if (!likedMovie) { if (resultEl) resultEl.innerText = 'Фильм не найден.'; return; }

  const likedSet = new Set(likedMovie.genres);
  const candidates = movies.filter(m => m.id !== likedMovie.id);

  const scored = candidates.map(m => ({
    ...m,
    score: jaccard(likedSet, new Set(m.genres))
  })).sort((a, b) => (b.score !== a.score ? b.score - a.score : a.title.localeCompare(b.title)));

  const top = scored.slice(0, 2);

  if (!top.length) { if (resultEl) resultEl.innerText = `Нет рекомендаций для «${likedMovie.title}».`; return; }

  // ВЫВОД С ПРОЦЕНТАМИ (один знак после запятой)
  const list = top.map(m => `${m.title} (${pct(m.score, 1)})`).join(', ');
  if (resultEl) {
    resultEl.innerText =
      likedSet.size === 0
        ? `У «${likedMovie.title}» нет жанров в данных, поэтому схожест
