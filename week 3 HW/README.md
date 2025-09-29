# Matrix Factorization Movie Recommender (Vanilla JS, MF v3)

**What’s new:**
- Robust data loading: tries to **fetch** `u.item` / `u.data`; if it fails (e.g., GitHub Pages issue),
  you can **upload** both files via the form at the top.
- UI gating: training is enabled only when data is present; user dropdown is populated from ratings.
- Everything else: biased MF with SGD, L2, per-epoch RMSE, progress bar, Top-N.

**How to run**
1. Put these files into your repo (root or `docs/`).  
2. Add `u.item` and `u.data` next to `index.html` **or** upload them in the UI.  
3. (GitHub Pages tip) Add a blank `.nojekyll` file if fetch acts weird.  
4. Open Pages → Train Model → pick a user → Get Recommendations.
