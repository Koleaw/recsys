/**
 * data.js — модуль загрузки и парсинга данных (u.item, u.data).
 * Экспортирует глобальные: movies, ratings и функции loadData, parseItemData, parseRatingData.
 */

let movies = [];
let ratings = [];

/** Асинхронно загружает u.item и u.data из той же папки, что index.html */
async function loadData() {
  const resultEl = document.getElementById('result');

  try {
    const itemResp = await fetch('u.item');
    if (!itemResp.ok) throw new Error(`Failed to load u.item (HTTP ${itemResp.status})`);
    const itemText = await itemResp.text();
    parseItemData(itemText);

    const dataResp = await fetch('u.data');
    if (!dataResp.ok) throw new Error(`Failed to load u.data (HTTP ${dataResp.status})`);
    const dataText = await dataResp.text();
    parseRatingData(dataText);

  } catch (err) {
    console.error(err);
    if (resultEl) resultEl.innerText = `Ошибка загрузки данных: ${err.message}`;
  }
}

/** Парсинг u.item (в конце 19 бинарных флагов жанров: unknown + 18 именованных) */
function parseItemData(text) {
  const genreNames = [
    "Action","Adventure","Animation","Children's","Comedy","Crime","Documentary",
    "Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance",
    "Sci-Fi","Thriller","War","Western"
  ];

  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    if (!line.trim()) continue;
    const parts = line.split('|');
    if (parts.length < 5) continue;

    const id = parseInt(parts[0], 10);
    const title = parts[1];

    const genres = [];
    const last19Start = parts.length - 19;
    for (let j = 0; j < 19; j++) {
      const flag = parts[last19Start + j];
      if (flag === '1' && j > 0) { // j=0 — это "unknown", пропускаем
        const g = genreNames[j - 1];
        if (g) genres.push(g);
      }
    }
    movies.push({ id, title, genres });
  }
}

/** Парсинг u.data: userId \t itemId \t rating \t timestamp */
function parseRatingData(text) {
  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    if (!line.trim()) continue;
    const parts = line.split('\t');
    if (parts.length < 4) continue;

    const userId = parseInt(parts[0], 10);
    const itemId = parseInt(parts[1], 10);
    const rating = parseInt(parts[2], 10);
    const timestamp = parseInt(parts[3], 10);

    ratings.push({ userId, itemId, rating, timestamp });
  }
}

