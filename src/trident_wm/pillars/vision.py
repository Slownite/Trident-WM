from torch import nn
from trident_wm.constants import SEQ_LEN
import torch
from einops import rearrange

class Vision(nn.Module):
    def __init__(self, out_dims: int =256) -> None:
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.linear = nn.Sequential(
            nn.Linear(384, out_features=out_dims),
            nn.LayerNorm(out_dims)
        )
    def forward(self, x: torch.Tensor)->torch.Tensor:
        input = rearrange(x, 'b s c h w -> (b s) c h w')
        dino_latent_space = self.backbone(input)
        latent_space = self.linear(dino_latent_space)
        latent = rearrange(latent_space, '(b s) v -> b s v', s=SEQ_LEN)
        return latent
