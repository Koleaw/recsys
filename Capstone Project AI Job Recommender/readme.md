Two-Tower Candidate-Job Recommender System
A complete two-tower (dual-encoder) deep learning recommender system for matching candidates to job positions, built according to the specifications in the instruction.md.

Architecture Overview
Two-Tower Model
Candidate Tower: Encodes candidate features (education, experience, skills, languages) into a fixed-dimensional vector
Job Tower: Encodes job requirements (description, required skills, education, languages) into a fixed-dimensional vector
Similarity: Cosine similarity between candidate and job embeddings for ranking
Feature Engineering Pipeline
Education Features (10-20 dims)

Degree level matching
Field and speciality matching (Jaccard similarity)
Degree level gap
Experience Features (20-40 dims)

Total years of experience
Years in required titles
Title similarity (embedding-based cosine similarity)
Recent role match
Language Features (10-20 dims)

Mandatory language coverage
Preferred language coverage
Language level gaps
Skills Features (20-50 dims)

Skill overlap count and ratio
Weighted skill match score
Skill embedding similarity
Location Features

Geodesic distance
Location match (considering relocation willingness and presence type)
Mandatory Criteria Features

Pass/fail for each critical requirement
Global Text Similarity (1-5 dims)

Embedding-based similarity between candidate experience and job description
Components
Core Modules
models.py: Data schemas for Candidate and JobPosition
feature_extraction.py: Feature extraction pipeline with text normalization, embeddings, and feature engineering
scoring.py: Interpretable layer-level scoring (S_edu, S_exp, S_lang, S_skill) and aggregated baseline score
two_tower_model.py: PyTorch implementation of the two-tower architecture
training.py: Training pipeline with InfoNCE loss for contrastive learning
inference.py: Inference workflow with hard filtering and recommendation generation
explainability.py: Explainability engine for human-readable match explanations
recommender_model.py: Main unified interface for the complete system
Installation
pip install -r requirements.txt
Usage
Basic Usage
from recommender_model import CandidateJobRecommender
from models import Candidate, JobPosition

# Initialize recommender
recommender = CandidateJobRecommender(device='cpu')

# Recommend candidates for a job
results = recommender.recommend_candidates(
    job=job_position,
    candidate_pool=candidate_list,
    top_k=10,
    include_explanations=True
)

# Recommend jobs for a candidate
job_results = recommender.recommend_jobs(
    candidate=candidate,
    job_pool=job_list,
    top_k=10,
    include_explanations=True
)
Training
from training import Trainer, CandidateJobDataset
from recommender_model import CandidateJobRecommender

# Initialize model
recommender = CandidateJobRecommender()

# Prepare dataset
train_dataset = CandidateJobDataset(
    candidates=candidates,
    jobs=jobs,
    positive_pairs=[(0, 0), (1, 1), ...],  # (candidate_idx, job_idx)
    negative_pairs=[(0, 1), (1, 0), ...]
)

# Train
trainer = Trainer(recommender.model_wrapper, device='cpu')
trainer.train(train_dataset, num_epochs=10)
trainer.save_model('checkpoint.pth')
Hard Filtering Rules
Before ML scoring, candidates are filtered based on:

Mandatory Criteria: Candidates failing any mandatory criteria are excluded
Mandatory Languages: Candidates not meeting mandatory language requirements are excluded
Location: For onsite jobs, candidates unwilling to relocate are excluded (unless same city/country)
Scoring Logic
Layer-Level Scores
S_edu: Education score (0-1)

0 if degree < required
Base 0.7 + 0.2 for field match + 0.1 for speciality match
S_exp: Experience score (0-1)

Weighted combination: 0.4 * total_years + 0.3 * role_years + 0.3 * title_similarity
S_lang: Language score (0-1)

0 if mandatory languages not met
0.7 * mandatory_coverage + 0.3 * preferred_coverage
S_skill: Skills score (0-1)

0 if mandatory skills missing
0.6 * overlap_ratio + 0.4 * weighted_match
Aggregated Baseline Score
S_base = 0.20 * S_edu + 0.40 * S_exp + 0.20 * S_lang + 0.20 * S_skill
The ML model learns richer interactions beyond this baseline.

Explainability
The system provides detailed explanations including:

Overall match score breakdown
Education, experience, language, and skills matching details
Strengths and weaknesses of the match
Reasons for filtering (if applicable)
Model Architecture
Candidate Features → Candidate Tower → Candidate Embedding (128-dim)
                                                      ↓
                                              Cosine Similarity
                                                      ↓
Job Features → Job Tower → Job Embedding (128-dim)
Both towers use:

Input: Structured features (150-dim) + Text embeddings (384-dim)
Hidden layers: [256, 128] with BatchNorm, ReLU, Dropout
Output: 128-dim normalized vector
Training Objective
Loss: InfoNCE (contrastive learning)
Positive pairs: (candidate, job) from applications/acceptances
Negative sampling: In-batch negatives
File Structure
.
├── models.py                  # Data schemas
├── feature_extraction.py     # Feature engineering
├── scoring.py                 # Scoring logic
├── two_tower_model.py         # Model architecture
├── training.py                # Training pipeline
├── inference.py               # Inference workflow
├── explainability.py          # Explainability engine
├── recommender_model.py       # Main interface
├── example_usage.py           # Usage examples
├── requirements.txt           # Dependencies
└── README.md                  # This file
Notes
The model uses pre-trained sentence transformers (all-MiniLM-L6-v2) for text embeddings
Taxonomy mappings (titles, skills) should be customized for your domain
Location coordinates should use a proper geocoding service in production
Mandatory criteria checking logic should be implemented based on your specific requirements
Future Enhancements
Cross-encoder re-ranker for fine-grained matching
Auxiliary losses for skill/language/title matching
Graph-based skill embeddings
Transformer over work history sequence
Online learning and model updates
