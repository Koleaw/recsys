"""
Feature extraction pipeline for Candidate-Job matching
"""
import re
import string
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import numpy as np
from geopy.distance import geodesic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from models import Candidate, JobPosition, Education, WorkExperience, LanguageProficiency, Skill


class TextNormalizer:
    """Text normalization utilities"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Lowercase, strip punctuation, remove extra whitespace"""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def remove_stopwords(text: str, stopwords: Optional[Set[str]] = None) -> str:
        """Remove stop words (optional)"""
        if stopwords is None:
            # Common English stopwords
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = text.split()
        return ' '.join([w for w in words if w not in stopwords])


class TaxonomyMapper:
    """Maps titles and skills to canonical IDs"""
    
    def __init__(self):
        # Example taxonomy mappings - in production, load from database/API
        self.title_taxonomy = {
            'software engineer': 'SE001',
            'data scientist': 'DS001',
            'data analyst': 'DA001',
            'machine learning engineer': 'MLE001',
            'product manager': 'PM001',
            'project manager': 'PJM001',
        }
        
        self.skill_taxonomy = {
            'python': 'SK001',
            'java': 'SK002',
            'javascript': 'SK003',
            'sql': 'SK004',
            'machine learning': 'SK005',
            'deep learning': 'SK006',
        }
    
    def map_title(self, title: str) -> str:
        """Map title to canonical ID"""
        normalized = TextNormalizer.normalize_text(title)
        return self.title_taxonomy.get(normalized, normalized)
    
    def map_skill(self, skill_name: str) -> str:
        """Map skill to canonical ID"""
        normalized = TextNormalizer.normalize_text(skill_name)
        return self.skill_taxonomy.get(normalized, normalized)


class EmbeddingExtractor:
    """Extract embeddings using pre-trained transformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize sentence transformer model"""
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode list of texts to embeddings"""
        if not texts:
            return np.zeros((1, self.embedding_dim))
        return self.model.encode(texts, convert_to_numpy=True)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text to embedding"""
        return self.encode([text])[0]


class DateProcessor:
    """Process dates and compute durations"""
    
    @staticmethod
    def compute_duration_years(start_date: datetime, end_date: Optional[datetime] = None) -> float:
        """Compute duration in years"""
        if end_date is None:
            end_date = datetime.now()
        delta = end_date - start_date
        return delta.days / 365.25
    
    @staticmethod
    def compute_total_experience(work_experiences: List[WorkExperience]) -> float:
        """Compute total years of experience"""
        total = 0.0
        for exp in work_experiences:
            total += DateProcessor.compute_duration_years(exp.start_date, exp.end_date)
        return total


class LocationProcessor:
    """Process location features"""
    
    # Example city coordinates - in production, use geocoding service
    CITY_COORDINATES = {
        'new york': (40.7128, -74.0060),
        'london': (51.5074, -0.1278),
        'paris': (48.8566, 2.3522),
        'tokyo': (35.6762, 139.6503),
        'san francisco': (37.7749, -122.4194),
    }
    
    @staticmethod
    def get_coordinates(city: str, country: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for city/country (simplified - use geocoding in production)"""
        key = f"{city.lower()}, {country.lower()}"
        # Try city first
        coords = LocationProcessor.CITY_COORDINATES.get(city.lower())
        if coords:
            return coords
        # Default fallback
        return (0.0, 0.0)
    
    @staticmethod
    def compute_geodesic_distance(city1: str, country1: str, city2: str, country2: str) -> float:
        """Compute geodesic distance in kilometers"""
        coords1 = LocationProcessor.get_coordinates(city1, country1)
        coords2 = LocationProcessor.get_coordinates(city2, country2)
        if coords1 and coords2:
            return geodesic(coords1, coords2).kilometers
        return float('inf')


class LanguageProcessor:
    """Process language proficiency levels"""
    
    LEVEL_ORDINAL = {
        'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4,
        'C1': 5, 'C2': 6, 'Native': 7
    }
    
    @staticmethod
    def level_to_ordinal(level: str) -> int:
        """Convert language level to ordinal"""
        return LanguageProcessor.LEVEL_ORDINAL.get(level.upper(), 0)
    
    @staticmethod
    def get_candidate_language_level(candidate: Candidate, language: str) -> int:
        """Get candidate's proficiency level for a language"""
        for lang_prof in candidate.languages:
            if lang_prof.language.lower() == language.lower():
                return LanguageProcessor.level_to_ordinal(lang_prof.level)
        return 0


class FeatureExtractor:
    """Main feature extraction class"""
    
    def __init__(self):
        self.normalizer = TextNormalizer()
        self.taxonomy = TaxonomyMapper()
        self.embedder = EmbeddingExtractor()
        self.date_processor = DateProcessor()
        self.location_processor = LocationProcessor()
        self.language_processor = LanguageProcessor()
    
    # Education Features
    def extract_education_features(self, candidate: Candidate, job: JobPosition) -> Dict[str, float]:
        """Extract education-related features"""
        features = {}
        
        # Get candidate's highest degree level
        degree_levels = {'high school': 1, 'bachelor': 2, 'master': 3, 'phd': 4}
        candidate_highest = 0
        candidate_fields = set()
        candidate_specialities = set()
        
        for edu in candidate.education:
            level_key = edu.level_of_education.lower()
            for key, val in degree_levels.items():
                if key in level_key:
                    candidate_highest = max(candidate_highest, val)
            candidate_fields.add(self.normalizer.normalize_text(edu.department))
            candidate_specialities.add(self.normalizer.normalize_text(edu.speciality))
        
        # Get required degree level
        required_level = 0
        if job.education_level:
            for key, val in degree_levels.items():
                if key in job.education_level.lower():
                    required_level = val
                    break
        
        features['candidate_highest_degree_level'] = float(candidate_highest)
        features['required_min_degree_level'] = float(required_level)
        features['degree_level_gap'] = float(candidate_highest - required_level)
        features['has_required_degree_level'] = 1.0 if candidate_highest >= required_level else 0.0
        
        # Field match score (Jaccard similarity)
        required_fields = set()
        if job.education_department:
            required_fields.add(self.normalizer.normalize_text(job.education_department))
        if job.education_speciality:
            required_fields.add(self.normalizer.normalize_text(job.education_speciality))
        
        if candidate_fields or required_fields:
            intersection = candidate_fields.intersection(required_fields)
            union = candidate_fields.union(required_fields)
            features['field_match_score'] = len(intersection) / len(union) if union else 0.0
        else:
            features['field_match_score'] = 0.0
        
        return features
    
    # Experience Features
    def extract_experience_features(self, candidate: Candidate, job: JobPosition) -> Dict[str, float]:
        """Extract work experience features"""
        features = {}
        
        # Total years of experience
        total_years = 0.0
        for exp in candidate.work_experience:
            end_date = exp.end_date if exp.end_date else datetime.now()
            total_years += self.date_processor.compute_duration_years(exp.start_date, end_date)
        
        features['total_years_experience'] = total_years
        
        # Title similarity using embeddings
        job_text = f"{job.title} {job.description}"
        exp_texts = [f"{exp.position} {exp.description}" for exp in candidate.work_experience]
        
        if exp_texts:
            job_emb = self.embedder.encode_single(job_text)
            exp_embs = self.embedder.encode(exp_texts)
            similarities = cosine_similarity([job_emb], exp_embs)[0]
            features['title_similarity_score'] = float(np.max(similarities))
        else:
            features['title_similarity_score'] = 0.0
        
        # Years in required titles
        required_titles = {self.taxonomy.map_title(job.title)}
        years_in_required = 0.0
        for exp in candidate.work_experience:
            mapped_title = self.taxonomy.map_title(exp.position)
            if mapped_title in required_titles:
                end_date = exp.end_date if exp.end_date else datetime.now()
                years_in_required += self.date_processor.compute_duration_years(exp.start_date, end_date)
        
        features['years_experience_in_required_titles'] = years_in_required
        
        # Experience approval
        required_years = self._parse_experience_level(job.level_of_experience)
        features['experience_approval'] = 1.0 if total_years >= required_years else 0.0
        
        # Recent role match
        if candidate.work_experience:
            most_recent = max(candidate.work_experience, key=lambda x: x.start_date)
            mapped_recent = self.taxonomy.map_title(most_recent.position)
            features['recent_role_match'] = 1.0 if mapped_recent in required_titles else 0.0
        else:
            features['recent_role_match'] = 0.0
        
        return features
    
    def _parse_experience_level(self, level_str: str) -> float:
        """Parse experience level string to years"""
        if not level_str:
            return 0.0
        level_lower = level_str.lower()
        if 'entry' in level_lower or 'junior' in level_lower:
            return 0.0
        elif 'mid' in level_lower or '2-5' in level_lower:
            return 3.0
        elif 'senior' in level_lower or '5+' in level_lower:
            return 5.0
        return 2.0  # default
    
    # Language Features
    def extract_language_features(self, candidate: Candidate, job: JobPosition) -> Dict[str, float]:
        """Extract language-related features"""
        features = {}
        
        mandatory_languages = []
        preferred_languages = []
        
        for lang_req in job.languages:
            if isinstance(lang_req, dict):
                lang_name = lang_req.get('languages', {}).get('language', '') if isinstance(lang_req.get('languages'), dict) else str(lang_req.get('languages', ''))
                critical_level = lang_req.get('Level_of_cretical', 0)
                if critical_level > 0:
                    mandatory_languages.append((lang_name, critical_level))
                else:
                    preferred_languages.append(lang_name)
        
        # Check mandatory languages
        mandatory_satisfied = 0
        language_gaps = []
        
        for lang_name, required_level in mandatory_languages:
            candidate_level = self.language_processor.get_candidate_language_level(candidate, lang_name)
            required_ordinal = self.language_processor.level_to_ordinal(str(int(required_level)))
            
            if candidate_level >= required_ordinal:
                mandatory_satisfied += 1
            language_gaps.append(float(candidate_level - required_ordinal))
        
        num_mandatory = len(mandatory_languages) if mandatory_languages else 1
        features['mandatory_language_coverage_ratio'] = mandatory_satisfied / num_mandatory if num_mandatory > 0 else 1.0
        features['all_mandatory_languages_ok'] = 1.0 if mandatory_satisfied == num_mandatory else 0.0
        
        # Preferred languages
        preferred_satisfied = 0
        for lang_name in preferred_languages:
            candidate_level = self.language_processor.get_candidate_language_level(candidate, lang_name)
            if candidate_level > 0:
                preferred_satisfied += 1
        
        num_preferred = len(preferred_languages) if preferred_languages else 1
        features['preferred_language_coverage_ratio'] = preferred_satisfied / num_preferred if num_preferred > 0 else 0.0
        
        # Average language gap
        features['avg_language_gap'] = float(np.mean(language_gaps)) if language_gaps else 0.0
        
        return features
    
    # Skills Features
    def extract_skills_features(self, candidate: Candidate, job: JobPosition) -> Dict[str, float]:
        """Extract skills-related features"""
        features = {}
        
        candidate_skill_ids = {self.taxonomy.map_skill(skill.skill_name) for skill in candidate.skills}
        job_skill_ids = {self.taxonomy.map_skill(skill.skill_name) for skill in job.skills}
        
        features['num_required_skills'] = float(len(job_skill_ids))
        
        # Overlap
        overlap = candidate_skill_ids.intersection(job_skill_ids)
        features['skill_overlap_count'] = float(len(overlap))
        features['skill_overlap_ratio'] = len(overlap) / len(job_skill_ids) if job_skill_ids else 0.0
        
        # Weighted skill match
        total_importance = sum(skill.importance or 1.0 for skill in job.skills)
        matched_importance = 0.0
        for skill in job.skills:
            mapped_skill = self.taxonomy.map_skill(skill.skill_name)
            if mapped_skill in candidate_skill_ids:
                matched_importance += skill.importance or 1.0
        
        features['weighted_skill_match_score'] = matched_importance / total_importance if total_importance > 0 else 0.0
        features['mandatory_skill_coverage_ratio'] = features['skill_overlap_ratio']  # Simplified
        
        # Skill embedding similarity
        if candidate.skills and job.skills:
            candidate_skill_texts = [skill.skill_name for skill in candidate.skills]
            job_skill_texts = [skill.skill_name for skill in job.skills]
            
            candidate_emb = np.mean(self.embedder.encode(candidate_skill_texts), axis=0)
            job_emb = np.mean(self.embedder.encode(job_skill_texts), axis=0)
            
            features['skill_embedding_similarity'] = float(cosine_similarity([candidate_emb], [job_emb])[0][0])
        else:
            features['skill_embedding_similarity'] = 0.0
        
        return features
    
    # Location Features
    def extract_location_features(self, candidate: Candidate, job: JobPosition) -> Dict[str, float]:
        """Extract location-related features"""
        features = {}
        
        # If job is online, location doesn't matter
        if job.presence_type.lower() == 'online':
            features['location_relevant'] = 0.0
            features['geodesic_distance'] = 0.0
            features['location_match'] = 1.0
            return features
        
        features['location_relevant'] = 1.0
        
        # Compute distance
        distance = self.location_processor.compute_geodesic_distance(
            candidate.city, candidate.country,
            job.city, job.country
        )
        features['geodesic_distance'] = distance
        
        # Check if same location
        same_city = candidate.city.lower() == job.city.lower()
        same_country = candidate.country.lower() == job.country.lower()
        
        # Location match logic
        if same_city and same_country:
            features['location_match'] = 1.0
        elif candidate.ready_to_relocate:
            features['location_match'] = 1.0
        else:
            features['location_match'] = 0.0
        
        return features
    
    # Mandatory Criteria Features
    def extract_mandatory_criteria_features(self, candidate: Candidate, job: JobPosition) -> Dict[str, float]:
        """Extract mandatory criteria features"""
        features = {}
        
        # Simplified: check if candidate meets critical requirements
        # In production, this would check actual criteria values
        passed_count = 0
        total_criteria = len(job.critical_requirements) if job.critical_requirements else 0
        
        # For now, assume all criteria are passed if no explicit failures
        # In production, implement actual criteria checking logic
        for req in job.critical_requirements:
            # Placeholder: would check actual candidate values against requirements
            passed_count += 1  # Simplified
        
        features['num_mandatory_criteria_passed'] = float(passed_count)
        features['mandatory_criteria_all_pass'] = 1.0 if passed_count == total_criteria else 0.0
        
        return features
    
    # Global text similarity
    def extract_global_text_similarity(self, candidate: Candidate, job: JobPosition) -> Dict[str, float]:
        """Extract global text embedding similarity"""
        features = {}
        
        # Aggregate candidate text
        candidate_texts = []
        for exp in candidate.work_experience:
            candidate_texts.append(f"{exp.position} {exp.description}")
        candidate_text = " ".join(candidate_texts)
        
        # Job text
        job_text = f"{job.title} {job.description} {job.qualifications or ''} {job.responsibilities or ''}"
        
        if candidate_text and job_text:
            candidate_emb = self.embedder.encode_single(candidate_text)
            job_emb = self.embedder.encode_single(job_text)
            features['global_text_similarity'] = float(cosine_similarity([candidate_emb], [job_emb])[0][0])
        else:
            features['global_text_similarity'] = 0.0
        
        return features
    
    def extract_all_features(self, candidate: Candidate, job: JobPosition) -> Dict[str, float]:
        """Extract all features for a candidate-job pair"""
        all_features = {}
        
        all_features.update(self.extract_education_features(candidate, job))
        all_features.update(self.extract_experience_features(candidate, job))
        all_features.update(self.extract_language_features(candidate, job))
        all_features.update(self.extract_skills_features(candidate, job))
        all_features.update(self.extract_location_features(candidate, job))
        all_features.update(self.extract_mandatory_criteria_features(candidate, job))
        all_features.update(self.extract_global_text_similarity(candidate, job))
        
        return all_features

