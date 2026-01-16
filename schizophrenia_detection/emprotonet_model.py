# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import torch
import torch.nn as nn


class PrototypeLayer(nn.Module):
    def __init__(self, num_prototypes, feature_dim):
        super(PrototypeLayer, self).__init__()
        self.num_prototypes = num_prototypes
        self.feature_dim = feature_dim

        # Learnable prototypes
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim) * 0.1)

        # Learnable importance weights for features
        self.feature_weights = nn.Parameter(torch.ones(feature_dim))

        # Learnable temperature for similarity
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # Apply feature weights
        weighted_x = x * self.feature_weights.unsqueeze(0)
        weighted_prototypes = self.prototypes * self.feature_weights.unsqueeze(0)

        # Compute weighted Euclidean distance
        x_expanded = weighted_x.unsqueeze(1)  # (batch, 1, features)
        p_expanded = weighted_prototypes.unsqueeze(0)  # (1, num_proto, features)

        distances = torch.sqrt(torch.sum((x_expanded - p_expanded) ** 2, dim=2) + 1e-6)

        # Convert distances to similarities
        similarities = torch.exp(-self.temperature * distances)

        return similarities, distances


class EMProtoNet(nn.Module):
    def __init__(self, input_dim, num_classes, num_prototypes=10):
        super(EMProtoNet, self).__init__()

        # Use LayerNorm instead of BatchNorm for better small batch behavior
        self.feature_norm = nn.LayerNorm(input_dim)

        # Feature extraction with residual connections
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),  # Reduced size for small dataset
            nn.LayerNorm(128),  # Use LayerNorm instead of BatchNorm
            nn.ELU(),
            nn.Dropout(0.2),  # Reduced dropout for small dataset

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ELU(),
        )

        # Prototype layer
        self.prototype_layer = PrototypeLayer(num_prototypes, 32)

        # Classification head with uncertainty estimation
        self.classifier = nn.Sequential(
            nn.Linear(num_prototypes, 16),  # Reduced size
            nn.ELU(),
            nn.Dropout(0.1),  # Reduced dropout
            nn.Linear(16, num_classes),
        )

        # Additional output for prediction confidence
        self.confidence_layer = nn.Sequential(
            nn.Linear(num_prototypes, 8),
            nn.ELU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_attention=False):
        # Feature normalization
        x_norm = self.feature_norm(x)

        combined = x_norm

        # Feature extraction
        features = self.feature_extractor(combined)

        # Prototype similarities
        similarities, distances = self.prototype_layer(features)

        # Classification
        logits = self.classifier(similarities)

        # Confidence estimation
        confidence = self.confidence_layer(similarities)

        if return_attention:
            return logits, similarities, distances, confidence
        return logits, similarities, distances, confidence


class AblationEMProtoNet(nn.Module):
    def __init__(self, input_dim, num_classes, num_prototypes=10):
        super(AblationEMProtoNet, self).__init__()

        # Use LayerNorm instead of BatchNorm for better small batch behavior
        self.feature_norm = nn.LayerNorm(input_dim)

        # Feature extraction with residual connections
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),  # Reduced size for small dataset
            nn.LayerNorm(128),  # Use LayerNorm instead of BatchNorm
            nn.ELU(),
            nn.Dropout(0.2),  # Reduced dropout for small dataset

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ELU(),
        )

        # Classification head with uncertainty estimation
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),  # Reduced size
            nn.ELU(),
            nn.Dropout(0.1),  # Reduced dropout
            nn.Linear(16, num_classes),
        )

        # Additional output for prediction confidence
        self.confidence_layer = nn.Sequential(
            nn.Linear(32, 8),
            nn.ELU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_attention=False):
        # Feature normalization
        x_norm = self.feature_norm(x)

        combined = x_norm

        # Feature extraction
        features = self.feature_extractor(combined)

        # Classification
        logits = self.classifier(features)

        # Confidence estimation
        confidence = self.confidence_layer(features)

        if return_attention:
            return logits, None, None, confidence
        return logits, None, None, confidence
