import torch
import wandb
import lightning as L
from torch.nn import functional as F
from einops import rearrange, asnumpy

class WorldSystem(L.LightningModule):
    def __init__(self, vision, memory, decoder, lr=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['vision', 'memory', 'decoder'])
        self.vision = vision
        self.memory = memory
        self.decoder = decoder
        self.lr = lr

    def forward(self, x):
        z = self.vision(x)
        h = self.memory(z)
        return self.decoder(h)

    def _common_step(self, batch):
        # Extract image sequence from LeRobot dict format
        video = batch["observation.image"]
        z = self.vision(video)
        
        # Teacher Forcing: Predict frame t+1 from t
        z_input = z[:, :-1, :]
        z_target = z[:, 1:, :].detach() 
        
        z_hat = self.memory(z_input)
        video_hat = self.decoder(z_hat)
        
        loss_latent = F.mse_loss(z_hat, z_target)
        loss_visual = F.mse_loss(video_hat, video[:, 1:])
        
        return loss_latent, loss_visual, video, video_hat

    def training_step(self, batch, batch_idx):
        loss_latent, loss_visual, _, _ = self._common_step(batch)
        total_loss = loss_latent + loss_visual
        
        self.log("train/total_loss", total_loss, prog_bar=True)
        self.log("train/latent_loss", loss_latent)
        self.log("train/visual_loss", loss_visual)
        return total_loss

    def validation_step(self, batch, batch_idx):
        loss_latent, loss_visual, video, video_hat = self._common_step(batch)
        
        self.log("val/visual_loss", loss_visual, prog_bar=True)
        self.log("val/latent_loss", loss_latent)

        if batch_idx == 0 and self.logger:
            self._log_video(video, video_hat, "val/prediction_vs_gt")

    def test_step(self, batch, batch_idx):
        # Evaluation on the unseen test split
        loss_latent, loss_visual, video, video_hat = self._common_step(batch)
        
        self.log("test/visual_loss", loss_visual)
        self.log("test/latent_loss", loss_latent)

        if batch_idx == 0 and self.logger:
            self._log_video(video, video_hat, "test/final_imagination")

    def _log_video(self, video, video_hat, key):
        # Prepare for W&B: (Batch, Time, Channels, Height, Width)
        sample_pred = video_hat[0].clamp(0, 1).detach().cpu()
        sample_gt = video[0, 1:].detach().cpu()
        
        # W&B Video expects (T, C, H, W) or (T, H, W, C) depending on internal ops, 
        # but concatenation is easiest on Width (dim=-1)
        comparison = torch.cat([sample_gt, sample_pred], dim=-1) 
        
        # Convert to NumPy (Frames, Channels, Height, Width) for wandb.Video
        video_array = (asnumpy(comparison) * 255).astype('uint8')
        
        self.logger.experiment.log({
            key: wandb.Video(video_array, fps=4, format="mp4")
        })

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.AdamW(trainable_params, lr=self.lr)
