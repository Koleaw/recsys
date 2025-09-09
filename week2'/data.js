/**
 * data.js — loading & parsing (u.item, u.data)
 * Exposes globals: movies, ratings, loadData(), parseItemData(), parseRatingData()
 */
'use strict';

let movies = [];
let ratings = [];

async function loadData() {
  const resultEl = document.getElementById('result');
  console.log('[data.js] loadData() start');

  try {
    // u.item
    const itemResp = await fetch('u.item', { cache: 'no-store' });
    if (!itemResp.ok) throw new Error(`Failed to load u.item (HTTP ${itemResp.status})`);
    const itemText = await itemResp.text();
    parseItemData(itemText);
    console.log(`[data.js] parsed movies: ${movies.length}`);

    // u.data
    const dataResp = await fetch('u.data', { cache: 'no-store' });
    if (!dataResp.ok) throw new Error(`Failed to load u.data (HTTP ${dataResp.status})`);
    const dataText = await dataResp.text();
    parseRatingData(dataText);
    console.log(`[data.js] parsed ratings: ${ratings.length}`);
  } catch (err) {
    console.error('[data.js] loadData() error:', err);
    if (resultEl) resultEl.innerText = `Error loading data: ${err.message}`;
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
    const start = parts.length - 19; // 19 genre flags: unknown + 18 named
    for (let j = 0; j < 19; j++) {
      const flag = parts[start + j];
      if (flag === '1' && j > 0) {           // skip "unknown"
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
    const parts = line.split('\t');
    if (parts.length < 4) continue;

    ratings.push({
      userId: parseInt(parts[0], 10),
      itemId: parseInt(parts[1], 10),
      rating: parseInt(parts[2], 10),
      timestamp: parseInt(parts[3], 10)
    });
  }
}
