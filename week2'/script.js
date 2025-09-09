/**
 * script.js — UI и логика рекомендаций.
 * Использует данные, загруженные в data.js (movies, ratings) и функцию loadData().
 */

/** J(A,B) = |A ∩ B| / |A ∪ B| — схожесть множеств жанров */
function jaccard(setA, setB) {
  if (setA.size === 0 && setB.size === 0) return 0;
  let intersection = 0;
  for (const v of setA) if (setB.has(v)) intersection++;
  const unionSize = new Set([...setA, ...setB]).size;
  return unionSize === 0 ? 0 : intersection / unionSize;
}

/** Форматирование 0..1 в проценты, например 0.5 → "50%" */
function pct(x, digits = 0) {
  return `${(x * 100).toFixed(digits)}%`;
}

window.onload = async () => {
  const resultEl = document.getElementById('result');
  if (resultEl) resultEl.innerText = 'Loading data...';

  await loadData();                 // ждём данные

  populateMoviesDropdown();

  if (resultEl) {
    resultEl.innerText = movies.length
      ? 'Data loaded. Please select a movie.'
      : 'Данные загружены, но фильмы не распознаны. Проверьте формат/путь u.item.';
  }
};

/** Заполнение выпадающего списка фильмами (по алфавиту) */
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

/** Главная функция: считает Jaccard и выводит ТОП-2 с процентами */
function getRecommendations() {
  const select = document.getElementById('movie-select');
  const resultEl = document.getElementById('result');

  // 1) выбранный ID
  const selectedVal = select ? select.value : '';
  if (!selectedVal) {
    if (resultEl) resultEl.innerText = 'Сначала выберите фильм.';
    return;
  }
  const selectedId

