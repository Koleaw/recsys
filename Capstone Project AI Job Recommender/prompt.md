Prompt: Build a Two-Tower Candidate–Job Recommender System

Instruction:
You are an expert ML architect. Build a two-tower (dual-encoder) recommender system that matches candidates to job positions. Use the following schemas, feature definitions, preprocessing rules, and scoring logic. Your output must describe:

The complete model architecture

Feature extraction pipeline

Candidate-tower encoder

Job-tower encoder

Pairwise scoring mechanism

Hard filtering logic

Training objectives

Inference workflow

Optional: explainability outputs

1. Input Schemas
1.1 Candidate schema
{
  "candidate_id": "string",
  "first_name": "string",
  "second_name": "string",
  "phone_number": "string",
  "date_of_birth": "DateTime?",
  "gender": "string?",
  "country": "string",
  "city": "string",
  "ready_to_relocate": "boolean",

  "education": [
    {
      "level_of_education": "string",
      "department": "string",
      "speciality": "string",
      "university": "string",
      "country": "string",
      "start_date": "DateTime",
      "end_date": "DateTime?"
    }
  ],

  "work_experience": [
    {
      "company_name": "string",
      "position": "string",
      "start_date": "DateTime",
      "end_date": "DateTime?",
      "is_present": false,
      "description": "string",
      "country": "string",
      "type_of_work": "string"
    }
  ],

  "languages": [
    {
      "languages": "LanguageProficiency"
    }
  ],

  "skills": [
    { "$ref": "Skill" }
  ]
}

1.2 Job position schema
{
  "organizationId": "string",
  "status": "string",
  "title": "string",
  "description": "string",
  "city": "string",
  "country": "string",
  "start_date": "DateTime",
  "end_date": "DateTime",
  "presence_type": "string",

  "languages": [
    {
      "languages": "LanguageProficiency",
      "Level_of_cretical": "number"
    }
  ],

  "level_of_experience": "string",
  "offer_description": "string",
  "number_of_volunteers": 0,
  "type_of_presence": "string",

  "education_level": "string?",
  "education_speciality": "string?",
  "experience": "string?",
  "education_department": "string?",
  "notes": "string?",
  "qualifications": "string?",
  "responsibilities": "string?",

  "skills": [
    { "$ref": "Skill" }
  ],

  "Cretical requirments": [
    {
      "id": "string",
      "requirements": "string",
      "Degree": "number"
    }
  ]
}

2. General Preprocessing Requirements

Implement:

Text normalization (lowercase, punctuation stripping, optional stop-word removal).

Mapping titles → canonical taxonomy IDs.

Mapping skills → standardized skill IDs.

Extracting transformer-based embeddings for:

work experience descriptions

job descriptions & responsibilities

Date normalization:

durations in years

recency features

Location processing:

geodesic distance

presence policy (onsite / online / hybrid)

relocation logic

Language normalization and mapping levels to ordinals.

3. Feature Engineering

Implement exactly the following sets of features.

3.1 Education Features

candidate_highest_degree_level

required_min_degree_level

degree_level_gap

has_required_degree_level

field_match_score (Jaccard between degree fields and required fields)

3.2 Experience Features

total_years_experience

title_similarity_score (embedding cosine similarity)

years_experience_in_required_titles

experience_approval flag

recent_role_match

3.3 Language Features

candidate_level – job_required_level

language_gap per language

mandatory_language_coverage_ratio

all_mandatory_languages_ok

preferred_language_coverage_ratio

3.4 Skills Features

num_required_skill

skill_overlap_count

skill_overlap_ratio

mandatory_skill_coverage_ratio

weighted_skill_match_score

skill_embedding_similarity (cosine)

3.5 Location Features

geodesic distance

presence_type logic:

if job is online → ignore location distance

if job is onsite and candidate can relocate → treat as valid match

3.6 Mandatory Criteria Features

MCi_pass for all mandatory items

num_mandatory_criteria_passed

mandatory_criteria_all_pass (hard filter)

3.7 Final Feature Vector

Combine all groups into approx:

10–20 dims (education)

20–40 dims (experience)

10–20 dims (language)

20–50 dims (skills)

binary + numeric location features

global text embedding similarity (1–5 dims)

4. Model Architecture: Two-Tower (Dual Encoder)
4.1 Candidate Tower

Encode:

candidate skill embeddings

aggregated work-experience embeddings

language features

education features

structured numeric features

optional: RNN/Transformer over work history

Output: fixed-dimensional candidate vector.

4.2 Job Tower

Encode:

job description embedding

required skills embedding

required language features

required education features

mandatory requirements

location/presence metadata

Output: fixed-dimensional job vector.

4.3 Interaction Layer

Use cosine similarity or dot-product for the retrieval score.
Optional: deep cross-layer after interaction for re-ranker.

5. Training Objective

Use:

contrastive learning (InfoNCE or triplet loss)

positive pairs: (candidate, job applied/accepted)

negative sampling: in-batch or cross-batch

Optional auxiliary losses:

skill-match regression

language-match regression

title-similarity regression

6. Hard Filtering Rules (Pre-Scoring)

Before computing model score:

If any mandatory job criteria fail → candidate excluded.

If job is onsite:

reject candidates unwilling to relocate unless same city/country.

If mandatory languages are not satisfied → exclude.

7. Scoring Logic (Optional interpretable layer)

Calculate:

S_edu

As defined:

0 if degree < required

else more weight for field & speciality match

S_exp

Combination of:

total years

years in required titles

title similarity

S_lang

0 if any mandatory language fails
Else blend mandatory + preferred coverage

S_skill

Mix of overlap ratio + weighted skill match
Penalty if mandatory skills missing

Final Explainable Score
S_base = 
  0.20 * S_edu +
  0.40 * S_exp +
  0.20 * S_lang +
  0.20 * S_skill


ML model can learn a richer version of these interactions.

8. Inference Workflow

For each job:

Apply hard filters to candidate pool.

Encode candidates via candidate tower.

Encode job via job tower.

Compute similarity score.

Return top-K candidates.

(Optional) Re-rank using cross-encoder for finer matching.

Produce explainability summaries:

which skills matched

which languages passed

experience alignment

reasons for filters/hard failures

9. Deliverables the model should produce

Architecture diagram (textual)

Feature pipeline description

Model training procedure

Inference and deployment spec

Explainability design