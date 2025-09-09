/**
 * data.js — загрузка и парсинг (u.item, u.data).
 */
let movies = [];
let ratings = [];

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
  } catch (e) {
    console.error(e);
    if (resultEl) resultEl.innerText = `Ошибка загрузки данных: ${e.message}`;
  }
}

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
    const start = parts.length - 19; // 19 флагов жанров
    for (let j = 0; j < 19; j++) {
      const flag = parts[start + j];
      if (flag === '1' && j > 0) { // j=0 — "unknown", пропускаем
        const g = genreNames[j - 1];
        if (g) genres.push(g);
      }
    }
    movies.push({ id, title, genres });
  }
}

function parseRatingData(text) {
  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    if (!line.trim()) continue;
    const parts = line.split('\t'); // табы
    if (parts.length < 4) continue;
    ratings.push({
      userId: parseInt(parts[0], 10),
      itemId: parseInt(parts[1], 10),
      rating: parseInt(parts[2], 10),
      timestamp: parseInt(parts[3], 10)
    });
  }
}
