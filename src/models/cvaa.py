"""
CVAA — Cross-lingual Visual Alignment Adapter
Reduces hallucination gap between Urdu and English prompting in LLaVA.

Architecture:
    Input: CLIP visual features (1024) + Urdu text embedding (4096) → concat (5120)
    Layer 1: Linear(5120, 512) + LayerNorm + GELU + Dropout
    Layer 2: Linear(512, 256)  + LayerNorm + GELU + Dropout
    Layer 3: Linear(256, 1024)
    Output: visual_features + layer3_output  ← residual connection
    Params: ~6.3M

Intuition:
    The adapter learns a correction vector in visual feature space.
    Given the Urdu question and the visual features, it asks:
    "what adjustment to the visual representation would make
     this Urdu prompt better aligned with what the image contains?"
    The residual ensures the original visual info is never lost.
"""

import torch
import torch.nn as nn


class CVAA(nn.Module):
    def __init__(
        self,
        visual_dim: int = 1024,     #CLIP output dim
        text_dim: int = 4096,       #Vicuna hidden state dim
        hidden1: int = 512,
        hidden2: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        input_dim = visual_dim + text_dim  #5120

        #layer 1 - compress from joint space
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        #layer 2 - further compression
        self.layer2 = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        #layer 3 - project back to visual space
        self.layer3 = nn.Linear(hidden2, visual_dim)

        #initialise layer3 near zero so residual starts as identity
        #this means at step 0 the model behaves like no adapter is present
        #and learns corrections gradually — much more stable training
        nn.init.zeros_(self.layer3.weight)
        nn.init.zeros_(self.layer3.bias)

    def forward(
        self,
        visual_features: torch.Tensor,   # (batch, 576, 1024)
        text_embedding:  torch.Tensor    # (batch, 4096)
    ) -> torch.Tensor:
        """
        Args:
            visual_features : CLIP patch embeddings before MLP projection
                              shape (batch, num_patches, visual_dim)
            text_embedding  : mean-pooled Urdu question hidden states
                              shape (batch, text_dim)
        Returns:
            aligned_features: visual_features + correction
                              shape (batch, num_patches, visual_dim)
        """
        batch, num_patches, visual_dim = visual_features.shape

        #pool visual features across patches for the adapter input
        #(batch, 1024)
        v_pooled = visual_features.mean(dim=1)

        #concatenate visual + text → (batch, 5120)
        combined = torch.cat([v_pooled, text_embedding], dim=-1)

        #forward through adapter layers
        x = self.layer1(combined)   # (batch, 512)
        x = self.layer2(x)          # (batch, 256)
        correction = self.layer3(x) # (batch, 1024)

        #expand correction to all patches and add as residual
        #(batch, 1024) → (batch, 576, 1024)
        correction = correction.unsqueeze(1).expand_as(visual_features)

        return visual_features + correction


if __name__ == "__main__":
    #sanity check
    model = CVAA()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    #dummy inputs matching real shapes
    visual = torch.randn(2, 576, 1024)  # batch=2, 576 patches, dim=1024
    text = torch.randn(2, 4096)       # batch=2, Vicuna hidden dim

    out = model(visual, text)
    print(f"Input  shape: {visual.shape}")
    print(f"Output shape: {out.shape}")   # should match input
    assert out.shape == visual.shape, "shape mismatch!"
    print("forward pass OK ✓")