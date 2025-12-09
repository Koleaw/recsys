"""
Example usage of the Candidate-Job Recommender System
"""
from datetime import datetime
from models import (
    Candidate, JobPosition, Education, WorkExperience,
    LanguageProficiency, Skill, CriticalRequirement
)
from recommender_model import CandidateJobRecommender


def create_example_candidate() -> Candidate:
    """Create an example candidate"""
    return Candidate(
        candidate_id="C001",
        first_name="John",
        second_name="Doe",
        phone_number="+1234567890",
        date_of_birth=datetime(1990, 1, 1),
        country="USA",
        city="New York",
        ready_to_relocate=True,
        education=[
            Education(
                level_of_education="Master",
                department="Computer Science",
                speciality="Machine Learning",
                university="MIT",
                country="USA",
                start_date=datetime(2012, 9, 1),
                end_date=datetime(2014, 6, 1)
            )
        ],
        work_experience=[
            WorkExperience(
                company_name="Tech Corp",
                position="Software Engineer",
                start_date=datetime(2014, 7, 1),
                end_date=datetime(2017, 6, 30),
                description="Developed machine learning models for recommendation systems",
                country="USA",
                type_of_work="Full-time"
            ),
            WorkExperience(
                company_name="AI Startup",
                position="Senior Data Scientist",
                start_date=datetime(2017, 7, 1),
                is_present=True,
                description="Lead ML projects and build production recommendation systems",
                country="USA",
                type_of_work="Full-time"
            )
        ],
        languages=[
            LanguageProficiency(language="English", level="Native"),
            LanguageProficiency(language="Spanish", level="B2")
        ],
        skills=[
            Skill(skill_id="SK001", skill_name="Python", importance=1.0),
            Skill(skill_id="SK002", skill_name="Java", importance=0.8),
            Skill(skill_id="SK005", skill_name="Machine Learning", importance=1.0),
            Skill(skill_id="SK006", skill_name="Deep Learning", importance=0.9),
        ]
    )


def create_example_job() -> JobPosition:
    """Create an example job position"""
    return JobPosition(
        organizationId="ORG001",
        status="Open",
        title="Senior Machine Learning Engineer",
        description="We are looking for an experienced ML engineer to build recommendation systems",
        city="San Francisco",
        country="USA",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        presence_type="hybrid",
        languages=[
            {
                "languages": LanguageProficiency(language="English", level="C1"),
                "Level_of_cretical": 5.0
            }
        ],
        level_of_experience="Senior (5+ years)",
        offer_description="Competitive salary and benefits",
        education_level="Master",
        education_speciality="Machine Learning",
        education_department="Computer Science",
        qualifications="5+ years of ML experience, strong Python skills",
        responsibilities="Design and implement ML models for production",
        skills=[
            Skill(skill_id="SK001", skill_name="Python", importance=1.0),
            Skill(skill_id="SK005", skill_name="Machine Learning", importance=1.0),
            Skill(skill_id="SK006", skill_name="Deep Learning", importance=0.9),
        ],
        critical_requirements=[
            CriticalRequirement(
                id="CR001",
                requirements="Must have 5+ years ML experience",
                Degree=1.0
            )
        ]
    )


def main():
    """Main example function"""
    print("Initializing Recommender System...")
    recommender = CandidateJobRecommender(device='cpu')
    
    # Create example data
    candidate = create_example_candidate()
    job = create_example_job()
    
    # Create a small candidate pool
    candidate_pool = [candidate]
    
    print("\n=== Recommending Candidates for Job ===")
    results = recommender.recommend_candidates(
        job=job,
        candidate_pool=candidate_pool,
        top_k=5,
        include_explanations=True
    )
    
    for i, (cand, score, explanation) in enumerate(results, 1):
        print(f"\n--- Candidate {i} (Score: {score:.4f}) ---")
        print(f"Name: {cand.first_name} {cand.second_name}")
        print(f"\nExplanation:")
        print(recommender.format_explanation(explanation))
    
    print("\n=== Recommending Jobs for Candidate ===")
    job_pool = [job]
    job_results = recommender.recommend_jobs(
        candidate=candidate,
        job_pool=job_pool,
        top_k=5,
        include_explanations=True
    )
    
    for i, (job_pos, score, explanation) in enumerate(job_results, 1):
        print(f"\n--- Job {i} (Score: {score:.4f}) ---")
        print(f"Title: {job_pos.title}")
        print(f"\nExplanation:")
        print(recommender.format_explanation(explanation))
    
    print("\n=== Single Match Score ===")
    score = recommender.get_match_score(candidate, job)
    print(f"Match Score: {score:.4f}")


if __name__ == "__main__":
    main()

