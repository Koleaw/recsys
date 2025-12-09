2.2 Candidate schema 

{
  "candidate_id": "string",
  "first_name": "string",
  "second_name": "string",
  "phone_number": "string",
  "date_of_birth": "DateTime?",
  "gender": "string?",
  "country": "string",
  "city": "string",
  "ready_to_relocate": "boolen",
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
    {
      "$ref": "Skill"
    }
  ]
}


2.3 Job position schema

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
      "languages": "LanguageProficiency"
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
    {
      "$ref": "Skill"
    }
  ],
  "Cretical requirments": [
    {
      "id": "string",
      "requirements": "string",
      "Degree": "number"
    }
  ]
}



3. The Feature of the model

We’ll generate pairwise features for (Candidate, Job) pairs, plus candidate-only and job-only features used in deep models or two-tower architectures.

3.1 General preprocessing

Normalize text:

Lowercase, strip punctuation, remove stop words where appropriate.

Map to taxonomies:

Titles → canonical title IDs (e.g. “Software Engineer”, “Data Scientist”).

Skills → skill IDs from internal skill graph.

Extract embeddings:

Sentence embeddings from resume ex: the description of the work experience with job descriptions and the requirements (e.g. pre-trained transformer).

Compute dates → durations in years

Location 

ready to relocate

3.2 Education features

Per (Candidate, Job):

candidate_highest_degree_level (ordinal: 1–4)

required_min_degree_level (ordinal)

degree_level_gap = candidate_highest_degree_level - required_min_degree_level

has_required_degree_level = 1 if candidate_level >= required_min_degree_level else 0

field_match_score:

Jaccard similarity between candidate degree fields and job required_education.fields.

3.3 Work experience features

From candidate history + job requirements:

total_years_experience

title_similarity_score:

Embedding similarity (cosine) between job title + description with the work experiences titles and thier description

years_experience_in_required_titles

Sum duration where candidate normalized_title in job.required_titles or similar titles via taxonomy.

experience_approval = total_years_experience_in_required_titel > Leve_of_Experience

recent_role_match:

Whether the most recent position aligns with job’s required titles/industry.

3.4 Languages features

For each required language:

candidate_level vs job_min_level.

language_gap = candidate_level - job_min_level.

Aggregated features:

mandatory_language_coverage_ratio = (# mandatory languages satisfied) / (# mandatory languages)

all_mandatory_languages_ok = 1 if coverage = 1 else 0.

preferred_language_coverage_ratio (for non-mandatory but preferred ones).

3.5 Skills features

Use normalized skills (from CV parsing + manual enrichment).

For job.required_skills:

num_required_skill

Candidate-job features:

skill_overlap_count = |candidate_skills ∩ job_required_skills|

skill_overlap_ratio = skill_overlap_count / num_required_skills

mandatory_skill_coverage_ratio

weighted_skill_match_score = sum(importance_i for each matched skill i) / sum(importance_i for all required skills)

Embeddings:

Aggregate candidate skill embeddings (average or graph-based) and job required skills embeddings; compute cosine similarity skill_embedding_similarity.

3.6 Location features

Derive from coordinates and location policy.

geodesic(candidate_location, job_location)

Check if the type of the presense is online then there is no need for taking the location into count 

Check if the candidate ready to relocate and the job is offline then the candidate will be taking into account

3.7 Mandatory criteria features

From:

job.mandatory_criteria_definitions (yse / no)

candidate.mandatory_criteria_values

For each MC1–MC4:

MCi_pass = boolean

MCi_reason_if_fail (for explainability only, not necessarily a numeric feature)

Optionally:

num_mandatory_criteria_passed

mandatory_criteria_all_pass = int(all(MCi_pass))

For scoring and ranking, candidates failing any active mandatory criteria are filtered out before ML scoring.

3.8 Final feature vector

Example layout:

Education features (10–20 dims)

Experience features (20–40 dims)

Language features (10–20 dims)

Skills features (20–50 dims)

Location features (binary relation yes or no)

Global text similarity (embedding-based, 1–5 dims)

Binary flags (mandatory criteria pass/fail, if you want model to learn edge cases, but still enforce hard filter externally)

4. Scoring & weighting logic

Use a two-stage strategy:

Rule-based hard filter (location + mandatory criteria) ( Binary relation )

ML-based scoring & ranking on remaining candidates

4.1 Layer-level scores

Before ML, you can compute interpretable sub-scores (used for explainability and as inputs):

Each layer score ∈ [0, 1]:

Education score S_edu 

If candidate_level < min_required_level → 0

Else:

base = 0.7

+0.2 if degree_field matches degree_field of the job.

+0.1 if degree_speciality matches the speciality of the job

*Experience score S_exp

exp_total_score = clamp(total_years / required_min_years_total, 0, 1)

exp_role_score = clamp(years_in_required_titles / required_min_years_in_role, 0, 1) (if defined)

title_similarity_score ∈ [0,1] mapped from cosine similarity.

Combine: S_exp = 0.4 * exp_total_score + 0.3 * exp_role_score + 0.3 * title_similarity_score.

Language score S_lang

If any mandatory language not met → 0

Else: S_lang = 0.7 * mandatory_language_coverage_ratio + 0.3 * preferred_language_coverage_ratio.

Skills score S_skill

If any mandatory skill missing → optionally set to 0 or strong penalty.

Else: S_skill = 0.6 * skill_overlap_ratio + 0.4 * weighted_skill_match_score.

4.2 Aggregated scoring (pre-ML baseline or fallback)

Define weights:

w_edu = 0.20

w_exp = 0.40

w_lang = 0.20

w_skill = 0.20

Then:

S_base = w_edu * S_edu
       + w_exp * S_exp
       + w_lang * S_lang
       + w_skill * S_skill


For production, the ML model learns these weights and interactions, but the above is a good interpretable baseline / fallback.