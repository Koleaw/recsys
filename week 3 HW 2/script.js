/**
 * script.js — TensorFlow.js Matrix Factorization (MF) model + UI logic.
 *
 * r_hat(u,i) = <P[u], Q[i]> — dot product of user & item embeddings.
 * Comments explain the important pieces of the model and data plumbing.
 */
'use strict';

let model = null;             // trained tf.Model
let userIndex = new Map();    // userId -> contiguous index
let movieIndex = new Map();   // movieId -> contiguous index

const $ = (id) => document.getElementById(id);
const log = (msg) => { const el = $('log'); if (el) { el.textContent += msg + '\\n'; el.scrollTop = el.scrollHeight; } };

window.onload = async () => {
  try {
    await loadData();     // from data.js
    populateDropdowns();
    $('result').textContent = 'Training model (TensorFlow.js)…';
    await trainModel();
    $('result').textContent = 'Model is ready. Select a user & movie, then click "Predict Rating".';
  } catch (e) {
    console.error(e);
  }
};

/** Fill <select>s and build ID->index maps */
function populateDropdowns() {
  userIndex = new Map(userIdList.map((id, idx) => [id, idx]));
  movieIndex = new Map(movieIdList.map((id, idx) => [id, idx]));

  const uSel = $('user-select');
  uSel.innerHTML = '<option selected disabled value=\"\">Select a user…</option>';
  for (const uid of userIdList) {
    const opt = document.createElement('option');
    opt.value = String(uid);
    opt.textContent = `User ${uid}`;
    uSel.appendChild(opt);
  }

  const mSel = $('movie-select');
  mSel.innerHTML = '<option selected disabled value=\"\">Select a movie…</option>';
  const movieById = new Map(movies.map(m => [m.id, m]));
  for (const mid of movieIdList) {
    const title = movieById.get(mid)?.title ?? `Movie ${mid}`;
    const opt = document.createElement('option');
    opt.value = String(mid);
    opt.textContent = title;
    mSel.appendChild(opt);
  }
}

/** Build the MF model:
 * - userInput/movieInput: integer IDs (shape [batch,1])
 * - Embedding layers: map IDs -> K-dim vectors (shape [batch,1,K])
 * - Flatten to [batch,K], then Dot along K to get [batch,1]
 */
function createModel(nUsers, nMovies, latentDim = 32) {
  const userInput = tf.input({shape: [1], dtype: 'int32', name: 'userInput'});
  const movieInput = tf.input({shape: [1], dtype: 'int32', name: 'movieInput'});

  const userEmbedding = tf.layers.embedding({
    inputDim: nUsers,
    outputDim: latentDim,
    embeddingsInitializer: tf.initializers.randomNormal({mean: 0, stddev: 0.05}),
    name: 'userEmbedding',
  }).apply(userInput);

  const movieEmbedding = tf.layers.embedding({
    inputDim: nMovies,
    outputDim: latentDim,
    embeddingsInitializer: tf.initializers.randomNormal({mean: 0, stddev: 0.05}),
    name: 'movieEmbedding',
  }).apply(movieInput);

  const userVec = tf.layers.flatten({name: 'userLatent'}).apply(userEmbedding);
  const movieVec = tf.layers.flatten({name: 'movieLatent'}).apply(movieEmbedding);

  const dot = tf.layers.dot({axes: 1, name: 'dot'}).apply([userVec, movieVec]);

  return tf.model({inputs: [userInput, movieInput], outputs: dot, name: 'MFModel'});
}

async function trainModel() {
  const latentDim = 32;
  model = createModel(numUsers, numMovies, latentDim);

  // Optimizer & loss as specified in the assignment
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError',
    metrics: ['mae'],
  });

  // Convert raw rating tuples into tensors
  const N = ratings.length;
  const u = new Int32Array(N);
  const m = new Int32Array(N);
  const y = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const r = ratings[i];
    u[i] = userIndex.get(r.userId);
    m[i] = movieIndex.get(r.itemId);
    y[i] = r.rating; // 1..5
  }

  const userTensor = tf.tensor2d(u, [N,1], 'int32');
  const movieTensor = tf.tensor2d(m, [N,1], 'int32');
  const yTensor = tf.tensor2d(y, [N,1], 'float32');

  $('result').textContent = 'Training…';
  log(`Training on ${N} samples (epochs=6, batchSize=256)`);

  await model.fit([userTensor, movieTensor], yTensor, {
    epochs: 6,
    batchSize: 256,
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        log(`Epoch ${epoch+1}: loss=${logs.loss.toFixed(4)}, mae=${logs.mae.toFixed(4)}`);
        await tf.nextFrame();
      }
    }
  });

  userTensor.dispose(); movieTensor.dispose(); yTensor.dispose();
}

/** Predict for selected (user, movie) */
async function predictRating() {
  if (!model) { $('result').textContent = 'Model is not trained.'; return; }

  const uId = parseInt(($('user-select').value || ''), 10);
  const mId = parseInt(($('movie-select').value || ''), 10);
  if (!uId || !mId) { $('result').textContent = 'Please select both user and movie.'; return; }

  const uIdx = userIndex.get(uId);
  const mIdx = movieIndex.get(mId);
  if (uIdx === undefined || mIdx === undefined) {
    $('result').textContent = 'User or movie not in training set.'; return;
  }

  const uT = tf.tensor2d([uIdx], [1,1], 'int32');
  const mT = tf.tensor2d([mIdx], [1,1], 'int32');

  const out = tf.tidy(() => model.predict([uT, mT]));
  const val = (await out.data())[0];
  uT.dispose(); mT.dispose(); out.dispose();

  const title = (movies.find(x => x.id === mId)?.title) || `Movie ${mId}`;
  const clipped = Math.max(1, Math.min(5, val));
  $('result').textContent = `Predicted rating for "${title}": ${clipped.toFixed(2)}/5`;
}
