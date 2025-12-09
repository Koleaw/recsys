"""
Scoring logic for candidate-job matching
"""
from typing import Dict, Tuple
import numpy as np

from feature_extraction import FeatureExtractor
from models import Candidate, JobPosition


class ScoringEngine:
    """Compute interpretable layer-level scores"""
    
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor

    def _get_weight_config(self, job: JobPosition) -> Dict[str, float]:
        """
        Return weights for S_edu, S_exp, S_lang, S_skill
        depending on job type (research vs engineering).
        """
        title = (job.title or "").lower()
        description = (job.description or "").lower()
        text = f"{title} {description}"

        research_keywords = ["research", "scientist", "r&d", "phd", "postdoc"]
        is_research = any(kw in text for kw in research_keywords)

        if is_research:
            # Research roles: education more important
            return {
                "w_edu": 0.35,
                "w_exp": 0.25,
                "w_lang": 0.15,
                "w_skill": 0.25,
            }
        else:
            # Engineering / default: experience more important
            return {
                "w_edu": 0.15,
                "w_exp": 0.45,
                "w_lang": 0.15,
                "w_skill": 0.25,
            }
    
    def compute_education_score(self, features: Dict[str, float]) -> float:
        """
        Education score S_edu
        If candidate_level < min_required_level → 0
        Else: base = 0.7 + 0.2 if field matches + 0.1 if speciality matches
        """
        if features['has_required_degree_level'] == 0.0:
            return 0.0
        
        base = 0.7
        if features.get('field_match_score', 0.0) > 0.5:  # Field match
            base += 0.2
        if features.get('field_match_score', 0.0) > 0.8:  # Speciality match (high field match)
            base += 0.1
        
        return min(base, 1.0)
    
    def compute_experience_score(self, features: Dict[str, float], job: JobPosition) -> float:
        """
        Experience score S_exp
        Combine: 0.4 * exp_total_score + 0.3 * exp_role_score + 0.3 * title_similarity_score
        """
        # Parse required years from job
        required_years = self.feature_extractor._parse_experience_level(job.level_of_experience)
        
        total_years = features['total_years_experience']
        exp_total_score = min(total_years / max(required_years, 1.0), 1.0) if required_years > 0 else 1.0
        
        years_in_role = features['years_experience_in_required_titles']
        exp_role_score = min(years_in_role / max(required_years, 1.0), 1.0) if required_years > 0 else 1.0
        
        title_similarity = features.get('title_similarity_score', 0.0)
        # Map cosine similarity [-1, 1] to [0, 1]
        title_similarity_normalized = (title_similarity + 1.0) / 2.0
        
        S_exp = 0.4 * exp_total_score + 0.3 * exp_role_score + 0.3 * title_similarity_normalized
        return min(max(S_exp, 0.0), 1.0)
    
    def compute_language_score(self, features: Dict[str, float]) -> float:
        """
        Language score S_lang
        If any mandatory language not met → 0
        Else: 0.7 * mandatory_coverage + 0.3 * preferred_coverage
        """
        if features.get('all_mandatory_languages_ok', 0.0) == 0.0:
            return 0.0
        
        mandatory_ratio = features.get('mandatory_language_coverage_ratio', 0.0)
        preferred_ratio = features.get('preferred_language_coverage_ratio', 0.0)
        
        S_lang = 0.7 * mandatory_ratio + 0.3 * preferred_ratio
        return min(max(S_lang, 0.0), 1.0)
    
    def compute_skills_score(self, features: Dict[str, float]) -> float:
        """
        Skills score S_skill
        If mandatory skills missing → 0 or penalty
        Else: 0.6 * overlap_ratio + 0.4 * weighted_match
        """
        # Check if mandatory skills are missing (simplified)
        if features.get('mandatory_skill_coverage_ratio', 0.0) < 0.5:
            return 0.0  # Strong penalty for missing mandatory skills
        
        overlap_ratio = features.get('skill_overlap_ratio', 0.0)
        weighted_match = features.get('weighted_skill_match_score', 0.0)
        
        S_skill = 0.6 * overlap_ratio + 0.4 * weighted_match
        return min(max(S_skill, 0.0), 1.0)
    
    def compute_aggregated_score(self, candidate: Candidate, job: JobPosition) -> Tuple[float, Dict[str, float]]:
        """
        Compute aggregated baseline score with hiring-type specific weights.
        """
        features = self.feature_extractor.extract_all_features(candidate, job)
        
        S_edu = self.compute_education_score(features)
        S_exp = self.compute_experience_score(features, job)
        S_lang = self.compute_language_score(features)
        S_skill = self.compute_skills_score(features)

        weights = self._get_weight_config(job)
        
        S_base = (
            weights["w_edu"] * S_edu
            + weights["w_exp"] * S_exp
            + weights["w_lang"] * S_lang
            + weights["w_skill"] * S_skill
        )
        
        layer_scores = {
            'S_edu': S_edu,
            'S_exp': S_exp,
            'S_lang': S_lang,
            'S_skill': S_skill,
            'S_base': S_base,
            'weights': weights,
        }
        
        return S_base, layer_scores


class HardFilter:
    """Hard filtering rules before ML scoring"""
    
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor
    
    def should_filter_out(self, candidate: Candidate, job: JobPosition) -> Tuple[bool, str]:
        """
        Check if candidate should be filtered out
        Returns: (should_filter, reason)
        """
        features = self.feature_extractor.extract_all_features(candidate, job)
        
        # 1. Check mandatory criteria
        if features.get('mandatory_criteria_all_pass', 0.0) == 0.0:
            return True, "Failed mandatory criteria"
        
        # 2. Check mandatory languages
        if features.get('all_mandatory_languages_ok', 0.0) == 0.0:
            return True, "Mandatory languages not satisfied"
        
        # 3. Check location (if job is onsite or hybrid)
        if job.presence_type.lower() != 'online':
            if features.get('location_match', 0.0) == 0.0:
                return True, "Location mismatch and candidate not willing to relocate"
        
        return False, "Passed all hard filters"
