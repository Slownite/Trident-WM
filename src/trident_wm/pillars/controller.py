import torch
from torch import nn
from einops import rearrange

class VisualDecoder(nn.Module):
    def __init__(self, input_latent_dim: int = 256) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=input_latent_dim, out_features=3136)
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(8, 4, 4, 2, 1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(4, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        x = rearrange(x, 'b s d -> (b s) d')
        x = self.linear(x)
        x = rearrange(x, 'bs (c h w) -> bs c h w', c=64, h=7, w=7)
        x = self.conv_transpose(x)
        return rearrange(x, '(b s) c h w -> b s c h w', b=b, s=s)
