"""
Explainability module for candidate-job matching
"""
from typing import Dict, List, Tuple
from models import Candidate, JobPosition
from feature_extraction import FeatureExtractor
from scoring import ScoringEngine


class ExplainabilityEngine:
    """Generate human-readable explanations for recommendations"""
    
    def __init__(self, feature_extractor: FeatureExtractor, scoring_engine: ScoringEngine):
        self.feature_extractor = feature_extractor
        self.scoring_engine = scoring_engine
    
    def generate_explanation(self, candidate: Candidate, job: JobPosition) -> Dict[str, any]:
        """
        Generate comprehensive explanation for a candidate-job match
        """
        features = self.feature_extractor.extract_all_features(candidate, job)
        _, layer_scores = self.scoring_engine.compute_aggregated_score(candidate, job)
        
        explanation = {
            'overall_score': layer_scores.get('S_base', 0.0),
            'breakdown': {
                'education': self._explain_education(candidate, job, features, layer_scores),
                'experience': self._explain_experience(candidate, job, features, layer_scores),
                'languages': self._explain_languages(candidate, job, features, layer_scores),
                'skills': self._explain_skills(candidate, job, features, layer_scores),
                'location': self._explain_location(candidate, job, features),
                'mandatory_criteria': self._explain_mandatory_criteria(candidate, job, features),
            },
            'strengths': self._identify_strengths(candidate, job, features, layer_scores),
            'weaknesses': self._identify_weaknesses(candidate, job, features, layer_scores),
        }
        
        return explanation
    
    def _explain_education(self, candidate: Candidate, job: JobPosition, features: Dict, layer_scores: Dict) -> Dict:
        """Explain education matching"""
        score = layer_scores.get('S_edu', 0.0)
        
        explanation = {
            'score': score,
            'candidate_highest_degree': features.get('candidate_highest_degree_level', 0),
            'required_degree': features.get('required_min_degree_level', 0),
            'meets_requirement': features.get('has_required_degree_level', 0) == 1.0,
            'field_match': features.get('field_match_score', 0),
        }
        
        if score == 0.0:
            explanation['reason'] = "Candidate does not meet minimum education requirement"
        elif score < 0.5:
            explanation['reason'] = "Education level meets requirement but field mismatch"
        else:
            explanation['reason'] = "Strong education match"
        
        return explanation
    
    def _explain_experience(self, candidate: Candidate, job: JobPosition, features: Dict, layer_scores: Dict) -> Dict:
        """Explain experience matching"""
        score = layer_scores.get('S_exp', 0.0)
        
        explanation = {
            'score': score,
            'total_years': features.get('total_years_experience', 0),
            'years_in_required_titles': features.get('years_experience_in_required_titles', 0),
            'title_similarity': features.get('title_similarity_score', 0),
            'recent_role_match': features.get('recent_role_match', 0) == 1.0,
        }
        
        if score < 0.3:
            explanation['reason'] = "Limited relevant experience"
        elif score < 0.7:
            explanation['reason'] = "Moderate experience match"
        else:
            explanation['reason'] = "Strong experience alignment"
        
        return explanation
    
    def _explain_languages(self, candidate: Candidate, job: JobPosition, features: Dict, layer_scores: Dict) -> Dict:
        """Explain language matching"""
        score = layer_scores.get('S_lang', 0.0)
        
        explanation = {
            'score': score,
            'all_mandatory_ok': features.get('all_mandatory_languages_ok', 0) == 1.0,
            'mandatory_coverage': features.get('mandatory_language_coverage_ratio', 0),
            'preferred_coverage': features.get('preferred_language_coverage_ratio', 0),
        }
        
        if score == 0.0:
            explanation['reason'] = "Mandatory languages not satisfied"
        elif score < 0.5:
            explanation['reason'] = "Some language requirements not fully met"
        else:
            explanation['reason'] = "Language requirements satisfied"
        
        return explanation
    
    def _explain_skills(self, candidate: Candidate, job: JobPosition, features: Dict, layer_scores: Dict) -> Dict:
        """Explain skills matching"""
        score = layer_scores.get('S_skill', 0.0)
        
        overlap_count = int(features.get('skill_overlap_count', 0))
        total_required = int(features.get('num_required_skills', 0))
        
        explanation = {
            'score': score,
            'matched_skills_count': overlap_count,
            'total_required_skills': total_required,
            'overlap_ratio': features.get('skill_overlap_ratio', 0),
            'weighted_match': features.get('weighted_skill_match_score', 0),
        }
        
        if score == 0.0:
            explanation['reason'] = "Critical skills missing"
        elif score < 0.5:
            explanation['reason'] = f"Only {overlap_count}/{total_required} required skills matched"
        else:
            explanation['reason'] = f"Strong skills match: {overlap_count}/{total_required} skills"
        
        return explanation
    
    def _explain_location(self, candidate: Candidate, job: JobPosition, features: Dict) -> Dict:
        """Explain location matching"""
        is_online = job.presence_type.lower() == 'online'
        location_match = features.get('location_match', 0) == 1.0
        distance = features.get('geodesic_distance', float('inf'))
        
        explanation = {
            'job_type': job.presence_type,
            'is_online': is_online,
            'location_match': location_match,
            'distance_km': distance if not is_online else 0,
            'candidate_relocates': candidate.ready_to_relocate,
        }
        
        if is_online:
            explanation['reason'] = "Online position - location not relevant"
        elif location_match:
            if distance < 50:
                explanation['reason'] = "Same city/region"
            else:
                explanation['reason'] = "Candidate willing to relocate"
        else:
            explanation['reason'] = "Location mismatch and candidate not willing to relocate"
        
        return explanation
    
    def _explain_mandatory_criteria(self, candidate: Candidate, job: JobPosition, features: Dict) -> Dict:
        """Explain mandatory criteria"""
        all_pass = features.get('mandatory_criteria_all_pass', 0) == 1.0
        passed_count = int(features.get('num_mandatory_criteria_passed', 0))
        total_count = len(job.critical_requirements) if job.critical_requirements else 0
        
        explanation = {
            'all_passed': all_pass,
            'passed_count': passed_count,
            'total_count': total_count,
        }
        
        if all_pass:
            explanation['reason'] = "All mandatory criteria satisfied"
        else:
            explanation['reason'] = f"Only {passed_count}/{total_count} mandatory criteria passed"
        
        return explanation
    
    def _identify_strengths(self, candidate: Candidate, job: JobPosition, features: Dict, layer_scores: Dict) -> List[str]:
        """Identify strengths of the match"""
        strengths = []
        
        if layer_scores.get('S_edu', 0) > 0.8:
            strengths.append("Excellent education match")
        if layer_scores.get('S_exp', 0) > 0.8:
            strengths.append("Strong relevant experience")
        if layer_scores.get('S_skill', 0) > 0.7:
            strengths.append("High skills overlap")
        if features.get('title_similarity_score', 0) > 0.7:
            strengths.append("Very similar role experience")
        if features.get('recent_role_match', 0) == 1.0:
            strengths.append("Current role aligns with job")
        
        return strengths
    
    def _identify_weaknesses(self, candidate: Candidate, job: JobPosition, features: Dict, layer_scores: Dict) -> List[str]:
        """Identify weaknesses of the match"""
        weaknesses = []
        
        if layer_scores.get('S_edu', 0) < 0.5:
            weaknesses.append("Education level or field mismatch")
        if layer_scores.get('S_exp', 0) < 0.5:
            weaknesses.append("Limited relevant experience")
        if layer_scores.get('S_lang', 0) < 0.5:
            weaknesses.append("Language requirements not fully met")
        if layer_scores.get('S_skill', 0) < 0.5:
            weaknesses.append("Skills gap")
        if features.get('location_match', 0) == 0 and job.presence_type.lower() != 'online':
            weaknesses.append("Location constraint")
        
        return weaknesses
    
    def format_explanation(self, explanation: Dict) -> str:
        """Format explanation as human-readable text"""
        lines = []
        lines.append(f"Overall Match Score: {explanation['overall_score']:.2%}")
        lines.append("")
        
        lines.append("=== Breakdown ===")
        for category, details in explanation['breakdown'].items():
            lines.append(f"\n{category.upper()}:")
            if isinstance(details, dict):
                for key, value in details.items():
                    if key != 'reason':
                        lines.append(f"  {key}: {value}")
                if 'reason' in details:
                    lines.append(f"  -> {details['reason']}")
        
        if explanation['strengths']:
            lines.append("\n=== Strengths ===")
            for strength in explanation['strengths']:
                lines.append(f"  + {strength}")
        
        if explanation['weaknesses']:
            lines.append("\n=== Areas for Improvement ===")
            for weakness in explanation['weaknesses']:
                lines.append(f"  - {weakness}")
        
        return "\n".join(lines)

