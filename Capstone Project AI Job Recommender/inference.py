"""
Inference workflow for candidate-job matching
"""
from typing import List, Tuple, Dict, Optional
import torch
import numpy as np
from tqdm import tqdm

from models import Candidate, JobPosition
from feature_extraction import FeatureExtractor
from scoring import HardFilter, ScoringEngine
from two_tower_model import ModelWrapper


class RecommenderSystem:
    """Complete recommender system with inference workflow"""
    
    def __init__(self, 
                 model: ModelWrapper,
                 feature_extractor: FeatureExtractor,
                 hard_filter: HardFilter,
                 scoring_engine: Optional[ScoringEngine] = None):
        self.model = model
        self.feature_extractor = feature_extractor
        self.hard_filter = hard_filter
        self.scoring_engine = scoring_engine or ScoringEngine(feature_extractor)
    
    def recommend_candidates_for_job(self, 
                                     job: JobPosition,
                                     candidate_pool: List[Candidate],
                                     top_k: int = 10,
                                     apply_hard_filter: bool = True) -> List[Tuple[Candidate, float, Dict]]:
        """
        Recommend top-K candidates for a job
        
        Returns:
            List of (candidate, score, explanation_dict) tuples, sorted by score descending
        """
        # Step 1: Apply hard filters
        if apply_hard_filter:
            filtered_candidates = []
            for candidate in candidate_pool:
                should_filter, reason = self.hard_filter.should_filter_out(candidate, job)
                if not should_filter:
                    filtered_candidates.append(candidate)
        else:
            filtered_candidates = candidate_pool
        
        if not filtered_candidates:
            return []
        
        # Step 2: Encode job
        self.model.model.eval()
        job_feat, job_emb = self.model.prepare_features(filtered_candidates[0], job)  # Dummy candidate for feature extraction
        job_feat_t = torch.FloatTensor(job_feat).unsqueeze(0)
        job_emb_t = torch.FloatTensor(job_emb).unsqueeze(0)
        
        with torch.no_grad():
            job_repr = self.model.model.encode_job(job_feat_t, job_emb_t)
        
        # Step 3: Encode all candidates and compute similarities
        results = []
        
        with torch.no_grad():
            for candidate in tqdm(filtered_candidates, desc="Encoding candidates"):
                # Prepare candidate features
                candidate_feat, candidate_emb = self.model.prepare_features(candidate, job)
                candidate_feat_t = torch.FloatTensor(candidate_feat).unsqueeze(0)
                candidate_emb_t = torch.FloatTensor(candidate_emb).unsqueeze(0)
                
                # Encode candidate
                candidate_repr = self.model.model.encode_candidate(candidate_feat_t, candidate_emb_t)
                
                # Compute similarity (cosine similarity since vectors are normalized)
                score = torch.sum(candidate_repr * job_repr, dim=1).item()
                
                # Generate explanation
                explanation = self._generate_explanation(candidate, job, score)
                
                results.append((candidate, score, explanation))
        
        # Step 4: Sort by score and return top-K
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def recommend_jobs_for_candidate(self,
                                     candidate: Candidate,
                                     job_pool: List[JobPosition],
                                     top_k: int = 10,
                                     apply_hard_filter: bool = True) -> List[Tuple[JobPosition, float, Dict]]:
        """
        Recommend top-K jobs for a candidate
        
        Returns:
            List of (job, score, explanation_dict) tuples, sorted by score descending
        """
        # Step 1: Apply hard filters
        if apply_hard_filter:
            filtered_jobs = []
            for job in job_pool:
                should_filter, reason = self.hard_filter.should_filter_out(candidate, job)
                if not should_filter:
                    filtered_jobs.append(job)
        else:
            filtered_jobs = job_pool
        
        if not filtered_jobs:
            return []
        
        # Step 2: Encode candidate
        self.model.model.eval()
        candidate_feat, candidate_emb = self.model.prepare_features(candidate, filtered_jobs[0])  # Dummy job
        candidate_feat_t = torch.FloatTensor(candidate_feat).unsqueeze(0)
        candidate_emb_t = torch.FloatTensor(candidate_emb).unsqueeze(0)
        
        with torch.no_grad():
            candidate_repr = self.model.model.encode_candidate(candidate_feat_t, candidate_emb_t)
        
        # Step 3: Encode all jobs and compute similarities
        results = []
        
        with torch.no_grad():
            for job in tqdm(filtered_jobs, desc="Encoding jobs"):
                # Prepare job features
                job_feat, job_emb = self.model.prepare_features(candidate, job)
                job_feat_t = torch.FloatTensor(job_feat).unsqueeze(0)
                job_emb_t = torch.FloatTensor(job_emb).unsqueeze(0)
                
                # Encode job
                job_repr = self.model.model.encode_job(job_feat_t, job_emb_t)
                
                # Compute similarity
                score = torch.sum(candidate_repr * job_repr, dim=1).item()
                
                # Generate explanation
                explanation = self._generate_explanation(candidate, job, score)
                
                results.append((job, score, explanation))
        
        # Step 4: Sort by score and return top-K
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _generate_explanation(self, candidate: Candidate, job: JobPosition, ml_score: float) -> Dict:
        """Generate explainability summary"""
        features = self.feature_extractor.extract_all_features(candidate, job)
        layer_scores = {}
        
        # Compute interpretable layer scores
        try:
            _, layer_scores = self.scoring_engine.compute_aggregated_score(candidate, job)
        except:
            pass
        
        # Extract key matching information
        explanation = {
            'ml_score': ml_score,
            'layer_scores': layer_scores,
            'matched_skills': self._get_matched_skills(candidate, job),
            'language_status': self._get_language_status(candidate, job, features),
            'experience_alignment': {
                'total_years': features.get('total_years_experience', 0),
                'years_in_required_titles': features.get('years_experience_in_required_titles', 0),
                'title_similarity': features.get('title_similarity_score', 0),
            },
            'education_match': {
                'has_required_degree': features.get('has_required_degree_level', 0) == 1.0,
                'field_match_score': features.get('field_match_score', 0),
            },
            'location_match': features.get('location_match', 0) == 1.0,
            'mandatory_criteria_passed': features.get('mandatory_criteria_all_pass', 0) == 1.0,
        }
        
        return explanation
    
    def _get_matched_skills(self, candidate: Candidate, job: JobPosition) -> List[str]:
        """Get list of matched skills"""
        candidate_skills = {self.feature_extractor.taxonomy.map_skill(s.skill_name) for s in candidate.skills}
        job_skills = {self.feature_extractor.taxonomy.map_skill(s.skill_name) for s in job.skills}
        matched = candidate_skills.intersection(job_skills)
        return list(matched)
    
    def _get_language_status(self, candidate: Candidate, job: JobPosition, features: Dict) -> Dict:
        """Get language matching status"""
        return {
            'all_mandatory_ok': features.get('all_mandatory_languages_ok', 0) == 1.0,
            'mandatory_coverage': features.get('mandatory_language_coverage_ratio', 0),
            'preferred_coverage': features.get('preferred_language_coverage_ratio', 0),
        }

