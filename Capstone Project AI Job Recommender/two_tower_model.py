"""
Two-Tower (Dual Encoder) Model Architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np

from feature_extraction import FeatureExtractor
from models import Candidate, JobPosition


class CandidateTower(nn.Module):
    """Candidate encoder tower"""
    
    def __init__(self, 
                 feature_dim: int = 150,
                 embedding_dim: int = 384,
                 hidden_dims: List[int] = [256, 128],
                 output_dim: int = 128):
        super(CandidateTower, self).__init__()
        
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        
        # Input projection
        input_dim = feature_dim + embedding_dim
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        
        # Layer normalization for final output
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, feature_vector: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_vector: [batch_size, feature_dim] - structured features
            embedding: [batch_size, embedding_dim] - text embeddings
        Returns:
            [batch_size, output_dim] - candidate representation
        """
        # Concatenate features and embeddings
        x = torch.cat([feature_vector, embedding], dim=1)
        
        # Pass through MLP
        output = self.mlp(x)
        
        # Layer normalization
        output = self.layer_norm(output)
        
        # L2 normalize for cosine similarity
        output = F.normalize(output, p=2, dim=1)
        
        return output


class JobTower(nn.Module):
    """Job encoder tower"""
    
    def __init__(self,
                 feature_dim: int = 150,
                 embedding_dim: int = 384,
                 hidden_dims: List[int] = [256, 128],
                 output_dim: int = 128):
        super(JobTower, self).__init__()
        
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        
        # Input projection
        input_dim = feature_dim + embedding_dim
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        
        # Layer normalization for final output
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, feature_vector: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_vector: [batch_size, feature_dim] - structured features
            embedding: [batch_size, embedding_dim] - text embeddings
        Returns:
            [batch_size, output_dim] - job representation
        """
        # Concatenate features and embeddings
        x = torch.cat([feature_vector, embedding], dim=1)
        
        # Pass through MLP
        output = self.mlp(x)
        
        # Layer normalization
        output = self.layer_norm(output)
        
        # L2 normalize for cosine similarity
        output = F.normalize(output, p=2, dim=1)
        
        return output


class TwoTowerModel(nn.Module):
    """Complete two-tower model"""
    
    def __init__(self,
                 feature_dim: int = 150,
                 embedding_dim: int = 384,
                 hidden_dims: List[int] = [256, 128],
                 output_dim: int = 128):
        super(TwoTowerModel, self).__init__()
        
        self.candidate_tower = CandidateTower(feature_dim, embedding_dim, hidden_dims, output_dim)
        self.job_tower = JobTower(feature_dim, embedding_dim, hidden_dims, output_dim)
        self.output_dim = output_dim
    
    def forward(self, 
                candidate_features: torch.Tensor,
                candidate_embeddings: torch.Tensor,
                job_features: torch.Tensor,
                job_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Returns cosine similarity scores
        """
        candidate_repr = self.candidate_tower(candidate_features, candidate_embeddings)
        job_repr = self.job_tower(job_features, job_embeddings)
        
        # Cosine similarity (dot product since vectors are normalized)
        scores = torch.sum(candidate_repr * job_repr, dim=1)
        
        return scores
    
    def encode_candidate(self, candidate_features: torch.Tensor, candidate_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode candidate to representation"""
        return self.candidate_tower(candidate_features, candidate_embeddings)
    
    def encode_job(self, job_features: torch.Tensor, job_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode job to representation"""
        return self.job_tower(job_features, job_embeddings)


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning"""
    
    def __init__(self, temperature: float = 0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, candidate_repr: torch.Tensor, job_repr: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss
        Args:
            candidate_repr: [batch_size, output_dim]
            job_repr: [batch_size, output_dim]
        Returns:
            scalar loss
        """
        batch_size = candidate_repr.size(0)
        
        # Compute similarity matrix
        # Since vectors are normalized, dot product = cosine similarity
        similarity_matrix = torch.matmul(candidate_repr, job_repr.t()) / self.temperature
        
        # Positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=candidate_repr.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class ModelWrapper:
    """Wrapper class for the complete model with feature extraction"""
    
    def __init__(self, 
                 feature_extractor: FeatureExtractor,
                 feature_dim: int = 150,
                 embedding_dim: int = 384,
                 hidden_dims: List[int] = [256, 128],
                 output_dim: int = 128):
        self.feature_extractor = feature_extractor
        self.model = TwoTowerModel(feature_dim, embedding_dim, hidden_dims, output_dim)
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
    
    def prepare_features(self, candidate: Candidate, job: JobPosition) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature vector and embedding for a candidate-job pair"""
        # Extract structured features
        features_dict = self.feature_extractor.extract_all_features(candidate, job)
        
        # Convert to numpy array (ordered)
        feature_keys = sorted(features_dict.keys())
        feature_vector = np.array([features_dict[k] for k in feature_keys])
        
        # Pad or truncate to feature_dim
        if len(feature_vector) < self.feature_dim:
            feature_vector = np.pad(feature_vector, (0, self.feature_dim - len(feature_vector)))
        else:
            feature_vector = feature_vector[:self.feature_dim]
        
        # Extract text embedding
        candidate_texts = []
        for exp in candidate.work_experience:
            candidate_texts.append(f"{exp.position} {exp.description}")
        candidate_text = " ".join(candidate_texts)
        
        job_text = f"{job.title} {job.description} {job.qualifications or ''} {job.responsibilities or ''}"
        
        combined_text = f"{candidate_text} {job_text}"
        embedding = self.feature_extractor.embedder.encode_single(combined_text)
        
        # Pad or truncate to embedding_dim
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dim]
        
        return feature_vector, embedding
    
    def predict_score(self, candidate: Candidate, job: JobPosition) -> float:
        """Predict matching score for a candidate-job pair"""
        self.model.eval()
        
        with torch.no_grad():
            # Prepare features
            candidate_feat, candidate_emb = self.prepare_features(candidate, job)
            job_feat, job_emb = self.prepare_features(candidate, job)  # Same for simplicity
            
            # Convert to tensors
            candidate_feat_t = torch.FloatTensor(candidate_feat).unsqueeze(0)
            candidate_emb_t = torch.FloatTensor(candidate_emb).unsqueeze(0)
            job_feat_t = torch.FloatTensor(job_feat).unsqueeze(0)
            job_emb_t = torch.FloatTensor(job_emb).unsqueeze(0)
            
            # Forward pass
            score = self.model(candidate_feat_t, candidate_emb_t, job_feat_t, job_emb_t)
            
            return score.item()

