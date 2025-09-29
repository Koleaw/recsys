/**
 * data.js — MovieLens data loading & parsing (u.item, u.data).
 * Exposes: global arrays `movies`, `ratings` and async `loadData()`.
 * - movies: [{ id, title, genres: string[] }]
 * - ratings: [{ userId, itemId, rating, timestamp }]
 */
'use strict';

let movies = [];
let ratings = [];

/** Load data files from the same directory as index.html */
async function loadData() {
  const status = document.getElementById('status');
  try {
    status && (status.textContent = 'Loading data files (u.item, u.data)...');

    // Fetch & parse u.item
    const itemResp = await fetch('./u.item', { cache: 'no-store' });
    if (!itemResp.ok) throw new Error(`Failed to load u.item (HTTP ${itemResp.status})`);
    const itemText = await itemResp.text();
    parseItemData(itemText);

    // Fetch & parse u.data
    const dataResp = await fetch('./u.data', { cache: 'no-store' });
    if (!dataResp.ok) throw new Error(`Failed to load u.data (HTTP ${dataResp.status})`);
    const dataText = await dataResp.text();
    parseRatingData(dataText);

    status && (status.textContent = 'Data loaded. Ready to train.');
  } catch (err) {
    console.error('[loadData] error:', err);
    status && (status.textContent = `Error: ${err.message}`);
    throw err;
  }
}

/** Parse u.item — last 19 fields are flags: [unknown, Action..Western]. Ignore 'unknown'. */
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
    const start = parts.length - 19;
    for (let j = 0; j < 19; j++) {
      const flag = parts[start + j];
      if (flag === '1' && j > 0) {
        const g = genreNames[j - 1];
        if (g) genres.push(g);
      }
    }
    movies.push({ id, title, genres });
  }
}

/** Parse u.data — tab-separated: userId itemId rating timestamp */
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
      timestamp: parseInt(parts[3], 10),
    });
  }
}
