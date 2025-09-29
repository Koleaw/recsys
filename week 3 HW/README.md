# Matrix Factorization Movie Recommender (Vanilla JS, MF v2)

**Files**
- `index.html`, `style.css` — UI
- `data.js` — load/parse `u.item` + `u.data` (MovieLens 100K format)
- `script.js` — trains a **biased matrix factorization** model in the browser with SGD
  - Prediction: `r_hat(u,i) = mu + b_u + b_i + p_u^T q_i`
  - L2 regularization, deterministic seed, train/val split, per-epoch RMSE
  - Top-N recommendations (option to exclude already rated items)

**How to run**
1. Put these files into your repo (root or `docs/`).
2. Add `u.item` and `u.data` right next to `index.html`.
3. Open the site (GitHub Pages). Click **Train Model** → then pick a user → **Get Recommendations**.

**Notes**
- Validation split is by ratings, not by users; you can set it to 0 to disable validation.
- Ratings are clipped to [0.5, 5.0] for display (MovieLens scale).
- Training yields to the browser regularly to keep UI responsive.
