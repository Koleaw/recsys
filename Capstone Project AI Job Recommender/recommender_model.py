"""
Main Recommender Model Class - Complete Two-Tower System
"""
import torch
from typing import List, Tuple, Dict, Optional

from models import Candidate, JobPosition
from feature_extraction import FeatureExtractor
from scoring import HardFilter, ScoringEngine
from two_tower_model import ModelWrapper
from inference import RecommenderSystem
from explainability import ExplainabilityEngine


class CandidateJobRecommender:
    """
    Complete Two-Tower Candidate-Job Recommender System
    
    This is the main class that provides a unified interface for:
    - Feature extraction
    - Hard filtering
    - Model inference
    - Recommendations
    - Explainability
    """
    
    def __init__(self,
                 feature_dim: int = 150,
                 embedding_dim: int = 384,
                 hidden_dims: List[int] = [256, 128],
                 output_dim: int = 128,
                 device: str = 'cpu'):
        """
        Initialize the recommender system
        
        Args:
            feature_dim: Dimension of structured features
            embedding_dim: Dimension of text embeddings
            hidden_dims: Hidden layer dimensions for towers
            output_dim: Output dimension of each tower
            device: Device to run model on ('cpu' or 'cuda')
        """
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.hard_filter = HardFilter(self.feature_extractor)
        self.scoring_engine = ScoringEngine(self.feature_extractor)
        self.model_wrapper = ModelWrapper(
            self.feature_extractor,
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        
        # Move model to device
        self.device = torch.device(device)
        self.model_wrapper.model.to(self.device)
        
        # Initialize recommender and explainability
        self.recommender = RecommenderSystem(
            self.model_wrapper,
            self.feature_extractor,
            self.hard_filter,
            self.scoring_engine
        )
        self.explainability = ExplainabilityEngine(
            self.feature_extractor,
            self.scoring_engine
        )
    
    def recommend_candidates(self,
                            job: JobPosition,
                            candidate_pool: List[Candidate],
                            top_k: int = 10,
                            apply_hard_filter: bool = True,
                            include_explanations: bool = True) -> List[Tuple[Candidate, float, Dict]]:
        """
        Recommend top-K candidates for a job
        
        Args:
            job: Job position to match against
            candidate_pool: List of candidate candidates
            top_k: Number of top candidates to return
            apply_hard_filter: Whether to apply hard filtering rules
            include_explanations: Whether to include detailed explanations
        
        Returns:
            List of (candidate, score, explanation_dict) tuples
        """
        results = self.recommender.recommend_candidates_for_job(
            job, candidate_pool, top_k, apply_hard_filter
        )
        
        if include_explanations:
            enhanced_results = []
            for candidate, score, basic_explanation in results:
                full_explanation = self.explainability.generate_explanation(candidate, job)
                enhanced_results.append((candidate, score, full_explanation))
            return enhanced_results
        
        return results
    
    def recommend_jobs(self,
                      candidate: Candidate,
                      job_pool: List[JobPosition],
                      top_k: int = 10,
                      apply_hard_filter: bool = True,
                      include_explanations: bool = True) -> List[Tuple[JobPosition, float, Dict]]:
        """
        Recommend top-K jobs for a candidate
        
        Args:
            candidate: Candidate to match jobs for
            job_pool: List of available jobs
            top_k: Number of top jobs to return
            apply_hard_filter: Whether to apply hard filtering rules
            include_explanations: Whether to include detailed explanations
        
        Returns:
            List of (job, score, explanation_dict) tuples
        """
        results = self.recommender.recommend_jobs_for_candidate(
            candidate, job_pool, top_k, apply_hard_filter
        )
        
        if include_explanations:
            enhanced_results = []
            for job, score, basic_explanation in results:
                full_explanation = self.explainability.generate_explanation(candidate, job)
                enhanced_results.append((job, score, full_explanation))
            return enhanced_results
        
        return results
    
    def get_match_score(self, candidate: Candidate, job: JobPosition) -> float:
        """Get matching score for a candidate-job pair"""
        return self.model_wrapper.predict_score(candidate, job)
    
    def get_explanation(self, candidate: Candidate, job: JobPosition) -> Dict:
        """Get detailed explanation for a candidate-job match"""
        return self.explainability.generate_explanation(candidate, job)
    
    def format_explanation(self, explanation: Dict) -> str:
        """Format explanation as human-readable text"""
        return self.explainability.format_explanation(explanation)
    
    def load_model(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model_wrapper.model.load_state_dict(checkpoint['model_state_dict'])
    
    def save_model(self, checkpoint_path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model_wrapper.model.state_dict(),
        }, checkpoint_path)
    
    def get_model(self):
        """Get the underlying PyTorch model"""
        return self.model_wrapper.model


if __name__ == "__main__":
    """
    Example usage when running this file directly
    """
    from datetime import datetime
    from models import (
        Education, WorkExperience, LanguageProficiency, Skill, CriticalRequirement
    )
    
    print("Initializing Recommender System...")
    recommender = CandidateJobRecommender(device='cpu')
    
    # Create example candidate
    candidate = Candidate(
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
    
    # Create example job
    job = JobPosition(
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
    
    # Create candidate pool
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

