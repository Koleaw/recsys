"""
Training pipeline for the two-tower model
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import numpy as np
from tqdm import tqdm

from models import Candidate, JobPosition
from two_tower_model import ModelWrapper, InfoNCELoss
from feature_extraction import FeatureExtractor
from scoring import ScoringEngine


class CandidateJobDataset(Dataset):
    """Dataset for candidate-job pairs"""
    
    def __init__(self, 
                 candidates: List[Candidate],
                 jobs: List[JobPosition],
                 positive_pairs: List[Tuple[int, int]],  # (candidate_idx, job_idx)
                 negative_pairs: Optional[List[Tuple[int, int]]] = None,
                 feature_extractor: Optional[FeatureExtractor] = None):
        self.candidates = candidates
        self.jobs = jobs
        self.positive_pairs = positive_pairs
        self.negative_pairs = negative_pairs or []
        self.feature_extractor = feature_extractor or FeatureExtractor()
        
        # Prepare all features
        self._prepare_features()
    
    def _prepare_features(self):
        """Pre-compute features for all candidates and jobs"""
        self.candidate_features = []
        self.candidate_embeddings = []
        self.job_features = []
        self.job_embeddings = []
        
        # Prepare candidate features
        for candidate in self.candidates:
            # Use dummy job for feature extraction (will be replaced in pairs)
            dummy_job = self.jobs[0] if self.jobs else None
            if dummy_job:
                feat, emb = self._extract_pair_features(candidate, dummy_job)
                self.candidate_features.append(feat)
                self.candidate_embeddings.append(emb)
        
        # Prepare job features
        for job in self.jobs:
            # Use dummy candidate for feature extraction
            dummy_candidate = self.candidates[0] if self.candidates else None
            if dummy_candidate:
                feat, emb = self._extract_pair_features(dummy_candidate, job)
                self.job_features.append(feat)
                self.job_embeddings.append(emb)
    
    def _extract_pair_features(self, candidate: Candidate, job: JobPosition) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features for a candidate-job pair"""
        features_dict = self.feature_extractor.extract_all_features(candidate, job)
        feature_keys = sorted(features_dict.keys())
        feature_vector = np.array([features_dict[k] for k in feature_keys])
        
        # Pad to fixed size
        feature_dim = 150
        if len(feature_vector) < feature_dim:
            feature_vector = np.pad(feature_vector, (0, feature_dim - len(feature_vector)))
        else:
            feature_vector = feature_vector[:feature_dim]
        
        # Extract embedding
        candidate_texts = []
        for exp in candidate.work_experience:
            candidate_texts.append(f"{exp.position} {exp.description}")
        candidate_text = " ".join(candidate_texts)
        
        job_text = f"{job.title} {job.description} {job.qualifications or ''} {job.responsibilities or ''}"
        combined_text = f"{candidate_text} {job_text}"
        
        embedding = self.feature_extractor.embedder.encode_single(combined_text)
        embedding_dim = 384
        if len(embedding) < embedding_dim:
            embedding = np.pad(embedding, (0, embedding_dim - len(embedding)))
        else:
            embedding = embedding[:embedding_dim]
        
        return feature_vector, embedding
    
    def __len__(self):
        return len(self.positive_pairs) + len(self.negative_pairs)
    
    def __getitem__(self, idx):
        if idx < len(self.positive_pairs):
            candidate_idx, job_idx = self.positive_pairs[idx]
            label = 1.0
        else:
            neg_idx = idx - len(self.positive_pairs)
            candidate_idx, job_idx = self.negative_pairs[neg_idx]
            label = 0.0
        
        candidate = self.candidates[candidate_idx]
        job = self.jobs[job_idx]
        
        # Extract actual pair features
        candidate_feat, candidate_emb = self._extract_pair_features(candidate, job)
        job_feat, job_emb = self._extract_pair_features(candidate, job)
        
        return {
            'candidate_features': torch.FloatTensor(candidate_feat),
            'candidate_embeddings': torch.FloatTensor(candidate_emb),
            'job_features': torch.FloatTensor(job_feat),
            'job_embeddings': torch.FloatTensor(job_emb),
            'label': torch.FloatTensor([label])
        }


class Trainer:
    """Training class for the two-tower model"""
    
    def __init__(self,
                 model: ModelWrapper,
                 learning_rate: float = 1e-4,
                 batch_size: int = 32,
                 device: str = 'cpu'):
        self.model = model
        self.device = torch.device(device)
        self.model.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.model.parameters(), lr=learning_rate)
        self.criterion = InfoNCELoss(temperature=0.07)
        self.batch_size = batch_size
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            candidate_feat = batch['candidate_features'].to(self.device)
            candidate_emb = batch['candidate_embeddings'].to(self.device)
            job_feat = batch['job_features'].to(self.device)
            job_emb = batch['job_embeddings'].to(self.device)
            
            # Forward pass
            candidate_repr = self.model.model.encode_candidate(candidate_feat, candidate_emb)
            job_repr = self.model.model.encode_job(job_feat, job_emb)
            
            # Compute loss
            loss = self.criterion(candidate_repr, job_repr)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, 
              train_dataset: CandidateJobDataset,
              num_epochs: int = 10,
              val_dataset: Optional[CandidateJobDataset] = None):
        """Train the model"""
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
            
            if val_dataset:
                val_loss = self.validate(val_dataset)
                print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")
    
    def validate(self, val_dataset: CandidateJobDataset) -> float:
        """Validate the model"""
        self.model.model.eval()
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                candidate_feat = batch['candidate_features'].to(self.device)
                candidate_emb = batch['candidate_embeddings'].to(self.device)
                job_feat = batch['job_features'].to(self.device)
                job_emb = batch['job_embeddings'].to(self.device)
                
                candidate_repr = self.model.model.encode_candidate(candidate_feat, candidate_emb)
                job_repr = self.model.model.encode_job(job_feat, job_emb)
                
                loss = self.criterion(candidate_repr, job_repr)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

