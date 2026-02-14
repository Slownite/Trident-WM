import torch
from torch import nn
from trident_wm.constants import SEQ_LEN

class Memory(nn.Module):
    def __init__(self, latent_dim: int = 256, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        self.pos_embedding = nn.Parameter(torch.zeros(1, SEQ_LEN, latent_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=latent_dim * 4,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(latent_dim, latent_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        h = self.transformer(x, mask=mask, is_causal=True)
        z_next_pred = self.output_head(h)
        
        return z_next_pred
