# Architecture Diagram

## Two-Tower Recommender System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                                   │
├─────────────────────────────────────────────────────────────────┤
│  Candidate Schema          │  Job Position Schema               │
│  - Education               │  - Title & Description            │
│  - Work Experience         │  - Required Skills                │
│  - Skills                  │  - Required Languages             │
│  - Languages               │  - Education Requirements         │
│  - Location                │  - Location & Presence Type        │
│  - Ready to Relocate       │  - Critical Requirements          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              FEATURE EXTRACTION PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│  1. Text Normalization                                          │
│     - Lowercase, punctuation removal                            │
│                                                                  │
│  2. Taxonomy Mapping                                            │
│     - Titles → Canonical IDs                                    │
│     - Skills → Skill IDs                                        │
│                                                                  │
│  3. Embedding Extraction                                        │
│     - Sentence Transformers (all-MiniLM-L6-v2)                 │
│     - Work experience descriptions                             │
│     - Job descriptions & requirements                           │
│                                                                  │
│  4. Feature Engineering                                         │
│     ├─ Education Features (10-20 dims)                         │
│     │  └─ Degree level, field match, gap                      │
│     ├─ Experience Features (20-40 dims)                         │
│     │  └─ Total years, role years, title similarity            │
│     ├─ Language Features (10-20 dims)                           │
│     │  └─ Coverage ratios, gaps                               │
│     ├─ Skills Features (20-50 dims)                             │
│     │  └─ Overlap, weighted match, embedding similarity        │
│     ├─ Location Features                                        │
│     │  └─ Geodesic distance, match flag                        │
│     ├─ Mandatory Criteria Features                              │
│     │  └─ Pass/fail flags                                      │
│     └─ Global Text Similarity (1-5 dims)                         │
│        └─ Embedding cosine similarity                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HARD FILTERING                                │
├─────────────────────────────────────────────────────────────────┤
│  ✓ Mandatory Criteria Check                                     │
│  ✓ Mandatory Languages Check                                    │
│  ✓ Location & Relocation Check                                  │
│                                                                  │
│  → Filtered Candidate Pool                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              TWO-TOWER MODEL ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐      ┌──────────────────────┐        │
│  │   CANDIDATE TOWER    │      │     JOB TOWER         │        │
│  ├──────────────────────┤      ├──────────────────────┤        │
│  │ Input:               │      │ Input:               │        │
│  │ - Features (150-dim) │      │ - Features (150-dim) │        │
│  │ - Embeddings (384)   │      │ - Embeddings (384)   │        │
│  │                      │      │                      │        │
│  │ Concatenate → 534    │      │ Concatenate → 534    │        │
│  │                      │      │                      │        │
│  │ Linear(534 → 256)    │      │ Linear(534 → 256)    │        │
│  │ BatchNorm + ReLU     │      │ BatchNorm + ReLU     │        │
│  │ Dropout(0.2)          │      │ Dropout(0.2)          │        │
│  │                      │      │                      │        │
│  │ Linear(256 → 128)    │      │ Linear(256 → 128)    │        │
│  │ BatchNorm + ReLU     │      │ BatchNorm + ReLU     │        │
│  │ Dropout(0.2)          │      │ Dropout(0.2)          │        │
│  │                      │      │                      │        │
│  │ Linear(128 → 128)    │      │ Linear(128 → 128)    │        │
│  │ LayerNorm            │      │ LayerNorm            │        │
│  │ L2 Normalize         │      │ L2 Normalize         │        │
│  │                      │      │                      │        │
│  │ Output: 128-dim      │      │ Output: 128-dim      │        │
│  │ normalized vector    │      │ normalized vector    │        │
│  └──────────────────────┘      └──────────────────────┘        │
│           │                              │                      │
│           └──────────┬───────────────────┘                      │
│                      ▼                                          │
│              Cosine Similarity                                  │
│           (Dot Product)                                         │
│                      │                                          │
│                      ▼                                          │
│              Match Score                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SCORING & RANKING                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer-Level Scores (for explainability):                       │
│  ┌──────────────────────────────────────────────┐              │
│  │ S_edu  = Education Score (0-1)                │              │
│  │ S_exp  = Experience Score (0-1)              │              │
│  │ S_lang = Language Score (0-1)                │              │
│  │ S_skill = Skills Score (0-1)                 │              │
│  └──────────────────────────────────────────────┘              │
│                                                                  │
│  Aggregated Baseline Score:                                     │
│  S_base = 0.20 * S_edu + 0.40 * S_exp +                        │
│           0.20 * S_lang + 0.20 * S_skill                       │
│                                                                  │
│  ML Model Score:                                                │
│  - Cosine similarity from two-tower embeddings                 │
│  - Learned interactions beyond baseline                        │
│                                                                  │
│  → Ranked Candidate-Job Pairs                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EXPLAINABILITY MODULE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Generate explanations:                                         │
│  - Overall match score breakdown                               │
│  - Education, experience, language, skills details             │
│  - Location and mandatory criteria status                       │
│  - Strengths and weaknesses                                    │
│  - Human-readable text format                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Training Flow

```
Positive Pairs (Candidate, Job)
         │
         ▼
┌────────────────────┐
│  Feature Extraction│
└────────────────────┘
         │
         ▼
┌────────────────────┐
│  Encode Candidates │
│  Encode Jobs       │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│  InfoNCE Loss      │
│  (Contrastive)     │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│  Backpropagation   │
│  Update Weights    │
└────────────────────┘
```

## Inference Flow

```
Job Query
   │
   ▼
┌────────────────────┐
│  Hard Filtering    │
│  (Mandatory checks)│
└────────────────────┘
   │
   ▼
┌────────────────────┐
│  Encode Job        │
└────────────────────┘
   │
   ▼
For each candidate:
   │
   ├─► Encode Candidate
   │
   ├─► Compute Similarity
   │
   └─► Generate Explanation
   │
   ▼
┌────────────────────┐
│  Rank & Return     │
│  Top-K Results     │
└────────────────────┘
```

## Feature Dimensions Summary

| Feature Category | Dimensions | Description |
|-----------------|------------|-------------|
| Education | 10-20 | Degree level, field match, gap |
| Experience | 20-40 | Years, role match, title similarity |
| Languages | 10-20 | Coverage ratios, gaps |
| Skills | 20-50 | Overlap, weighted match, embeddings |
| Location | Binary + numeric | Distance, match flag |
| Mandatory Criteria | Binary flags | Pass/fail per criterion |
| Global Text | 1-5 | Embedding similarity |
| **Total Structured** | **~150** | Combined feature vector |
| **Text Embeddings** | **384** | Sentence transformer output |
| **Tower Input** | **534** | Features + Embeddings |
| **Tower Output** | **128** | Normalized representation |

## Model Hyperparameters

- **Feature Dimension**: 150
- **Embedding Dimension**: 384 (sentence-transformers)
- **Hidden Layers**: [256, 128]
- **Output Dimension**: 128
- **Dropout Rate**: 0.2
- **Temperature (InfoNCE)**: 0.07
- **Learning Rate**: 1e-4
- **Batch Size**: 32

## Scoring Weights (Baseline)

- Education: 0.20
- Experience: 0.40
- Languages: 0.20
- Skills: 0.20

The ML model learns richer interactions and can adjust these weights based on data.

