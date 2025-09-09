/**
 * data.js – Data loading & parsing module.
 * Responsible ONLY for fetching and parsing local data files (u.item, u.data).
 * Exposes global variables `movies`, `ratings` and functions `loadData`, `parseItemData`, `parseRatingData`.
 */

// Global datasets
let movies = [];
let ratings = [];

/**
 * Load MovieLens-style data from local files.
 * - Fetches u.item and u.data from the same directory.
 * - Parses them into global `movies` and `ratings`.
 * - Displays a user-friendly error in #result if something goes wrong.
 */
async function loadData() {
  const resultEl = document.getElementById('result');

  try {
    // --- Load & parse items ---
    const itemResp = await fetch('u.item');
    if (!itemResp.ok) {
      throw new Error(`Failed to load u.item (HTTP ${itemResp.status})`);
    }
    const itemText = await itemResp.text();
    parseItemData(itemText);

    // --- Load & parse ratings ---
    const dataResp = await fetch('u.data');
    if (!dataResp.ok) {
      throw new Error(`Failed to load u.data (HTTP ${dataResp.status})`);
    }
    const dataText = await dataResp.text();
    parseRatingData(dataText);

  } catch (err) {
    console.error(err);
    if (resultEl) {
      resultEl.innerText = `Error loading data: ${err.message}`;
    }
  }
}

/**
 * Parse u.item text.
 * ML-100k has 19 binary genre flags at the end of each line: [unknown, Action..Western].
 * Spec requires defining the 18 named genres (Action → Western) and ignoring "unknown".
 * @param {string} text
 */
function parseItemData(text) {
  // 18 named genres (excluding 'unknown' by design)
  const genreNames = [
    "Action","Adventure","Animation","Children's","Comedy","Crime","Documentary",
    "Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance",
    "Sci-Fi","Thriller","War","Western"
  ];

  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    if (!line.trim()) continue;

    // u.item fields are | delimited
    const parts = line.split('|');
    if (parts.length < 5) continue; // malformed guard

    const id = parseInt(parts[0], 10);
    const title = parts[1];

    // The last 19 fields are genre flags. Index 0 is 'unknown' and must be ignored.
    const genres = [];
    const last19Start = parts.length - 19;

    for (let j = 0; j < 19; j++) {
      const flag = parts[last19Start + j];
      if (flag === '1') {
        // Skip 'unknown' (j === 0)
        if (j > 0) {
          const genreName = genreNames[j - 1]; // shift by one to align 18 named genres
          if (genreName) genres.push(genreName);
        }
      }
    }

    movies.push({ id, title, genres });
  }
}

/**
 * Parse u.data text.
 * Each line: userId \t itemId \t rating \t timestamp
 * @param {string} text
 */
function parseRatingData(text) {
  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    if (!line.trim()) continue;

    // u.data is tab-separated
    const parts = line.split('\t');
    if (parts.length < 4) continue;

    const userId = parseInt(parts[0], 10);
    const itemId = parseInt(parts[1], 10);
    const rating = parseInt(parts[2], 10);
    const timestamp = parseInt(parts[3], 10);

    ratings.push({ userId, itemId, rating, timestamp });
  }
}
