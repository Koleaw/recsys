"""
Data models for Candidate and Job Position schemas
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class Education:
    """Education entry schema"""
    level_of_education: str
    department: str
    speciality: str
    university: str
    country: str
    start_date: datetime
    end_date: Optional[datetime] = None


@dataclass
class WorkExperience:
    """Work experience entry schema"""
    company_name: str
    position: str
    start_date: datetime
    end_date: Optional[datetime] = None
    is_present: bool = False
    description: str = ""
    country: str = ""
    type_of_work: str = ""


@dataclass
class LanguageProficiency:
    """Language proficiency schema"""
    language: str
    level: str  # e.g., "A1", "A2", "B1", "B2", "C1", "C2", "Native"


@dataclass
class Skill:
    """Skill schema"""
    skill_id: str
    skill_name: str
    importance: Optional[float] = None  # For weighted matching


@dataclass
class CriticalRequirement:
    """Critical requirement schema"""
    id: str
    requirements: str
    Degree: float  # Importance/weight


@dataclass
class Candidate:
    """Candidate schema"""
    candidate_id: str
    first_name: str
    second_name: str
    phone_number: str
    date_of_birth: Optional[datetime] = None
    gender: Optional[str] = None
    country: str = ""
    city: str = ""
    ready_to_relocate: bool = False
    education: List[Education] = None
    work_experience: List[WorkExperience] = None
    languages: List[LanguageProficiency] = None
    skills: List[Skill] = None
    
    def __post_init__(self):
        if self.education is None:
            self.education = []
        if self.work_experience is None:
            self.work_experience = []
        if self.languages is None:
            self.languages = []
        if self.skills is None:
            self.skills = []


@dataclass
class JobPosition:
    """Job position schema"""
    organizationId: str
    status: str
    title: str
    description: str
    city: str
    country: str
    start_date: datetime
    end_date: datetime
    presence_type: str  # "online", "onsite", "hybrid"
    languages: List[Dict[str, Any]] = None  # [{"languages": LanguageProficiency, "Level_of_cretical": number}]
    level_of_experience: str = ""
    offer_description: str = ""
    number_of_volunteers: int = 0
    type_of_presence: str = ""
    education_level: Optional[str] = None
    education_speciality: Optional[str] = None
    experience: Optional[str] = None
    education_department: Optional[str] = None
    notes: Optional[str] = None
    qualifications: Optional[str] = None
    responsibilities: Optional[str] = None
    skills: List[Skill] = None
    critical_requirements: List[CriticalRequirement] = None
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = []
        if self.skills is None:
            self.skills = []
        if self.critical_requirements is None:
            self.critical_requirements = []

