/**
 * data.js — Load & parse MovieLens 100K files (u.item, u.data).
 * Exposes:
 *   - global arrays: `movies`, `ratings`
 *   - async function: `loadData()`
 *   - counts: `numUsers`, `numMovies`
 *   - identity lists: `userIdList`, `movieIdList` (sorted unique IDs)
 */
'use strict';

let movies = [];
let ratings = [];
let numUsers = 0;
let numMovies = 0;
let userIdList = [];
let movieIdList = [];

async function loadData() {
  const res = document.getElementById('result');
  try {
    res && (res.textContent = 'Loading data (u.item, u.data)…');

    const [itemResp, dataResp] = await Promise.all([
      fetch('./u.item', { cache: 'no-store' }),
      fetch('./u.data', { cache: 'no-store' }),
    ]);
    if (!itemResp.ok) throw new Error(`Failed to load u.item (HTTP ${itemResp.status})`);
    if (!dataResp.ok) throw new Error(`Failed to load u.data (HTTP ${dataResp.status})`);

    const [itemText, dataText] = await Promise.all([itemResp.text(), dataResp.text()]);
    parseItemData(itemText);
    parseRatingData(dataText);

    res && (res.textContent = `Data loaded: ${movies.length} movies, ${ratings.length} ratings.`);
  } catch (err) {
    console.error('[loadData] error:', err);
    res && (res.textContent = `Error while loading data: ${err.message}`);
    throw err;
  }
}

/** Parse u.item — last 19 fields are flags: [unknown, Action..Western]; we ignore 'unknown'. */
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
      if (flag === '1' && j > 0) { // skip 'unknown' at j=0
        const g = genreNames[j - 1];
        if (g) genres.push(g);
      }
    }
    movies.push({ id, title, genres });
  }
}

/** Parse u.data — TSV: userId  itemId  rating  timestamp */
function parseRatingData(text) {
  const uSet = new Set();
  const mSet = new Set();

  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    if (!line.trim()) continue;
    const parts = line.split('\t');
    if (parts.length < 4) continue;
    const userId = parseInt(parts[0], 10);
    const itemId = parseInt(parts[1], 10);
    const rating  = parseInt(parts[2], 10);
    const timestamp = parseInt(parts[3], 10);
    ratings.push({ userId, itemId, rating, timestamp });
    uSet.add(userId);
    mSet.add(itemId);
  }
  userIdList = Array.from(uSet).sort((a,b)=>a-b);
  movieIdList = Array.from(mSet).sort((a,b)=>a-b);
  numUsers = userIdList.length;
  numMovies = movieIdList.length;
}
