import click
import torch
import lightning as L
from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger

from trident_wm.pillars.vision import Vision
from trident_wm.pillars.memory import Memory
from trident_wm.pillars.controller import VisualDecoder
from trident_wm.system import WorldSystem
from trident_wm.datamodule import PushTDataModule

@click.group()
def main():
    """🔱 trident-WM: World Model Sprint (V-M-D)"""
    pass

@main.command()
@click.option('--config', type=click.Path(exists=True), required=True)
@click.option('--device', default=None, help="Override accelerator (cpu/gpu)")
def train(config, device):
    """Train the World Model using a YAML configuration"""
    torch.set_float32_matmul_precision('high')
    cfg = OmegaConf.load(config)
    accelerator = device if device else cfg.trainer.get('accelerator', 'auto')
    devices = cfg.trainer.get('devices', 'auto')

    # Pillar Initialization
    vision = Vision(out_dims=cfg.model.latent_dim)
    memory = Memory(latent_dim=cfg.model.latent_dim, nhead=cfg.model.nhead, num_layers=cfg.model.num_layers)
    decoder = VisualDecoder(input_latent_dim=cfg.model.latent_dim)

    system = WorldSystem(vision=vision, memory=memory, decoder=decoder, lr=cfg.model.lr)
    dm = PushTDataModule(repo_id=cfg.data.repo_id, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)

    logger = WandbLogger(project=cfg.wandb.project, name=cfg.wandb.name, config=OmegaConf.to_container(cfg, resolve=True))

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_kwargs.update({'accelerator': accelerator, 'devices': devices})
    
    trainer = L.Trainer(logger=logger, **trainer_kwargs)
    trainer.fit(system, datamodule=dm)

@main.command()
@click.option('--config', type=click.Path(exists=True), required=True)
@click.option('--checkpoint', type=click.Path(exists=True), required=True)
def evaluate(config, checkpoint):
    """Evaluate a saved model on the test dataset split"""
    cfg = OmegaConf.load(config)
    
    # 1. Reconstruct Architecture from Config
    vision = Vision(out_dims=cfg.model.latent_dim)
    memory = Memory(latent_dim=cfg.model.latent_dim, nhead=cfg.model.nhead, num_layers=cfg.model.num_layers)
    decoder = VisualDecoder(input_latent_dim=cfg.model.latent_dim)

    # 2. Load Weights from Checkpoint
    system = WorldSystem.load_from_checkpoint(checkpoint, vision=vision, memory=memory, decoder=decoder)
    
    # 3. Data (Auto-loads test split)
    dm = PushTDataModule(repo_id=cfg.data.repo_id, batch_size=cfg.data.batch_size)
    
    logger = WandbLogger(project=cfg.wandb.project, name=f"eval-{cfg.wandb.name}")
    trainer = L.Trainer(accelerator="auto", devices=1, logger=logger)
    
    click.echo(f"🧪 Running test evaluation on {checkpoint}...")
    trainer.test(system, datamodule=dm)

@main.command()
@click.option('--config', type=click.Path(exists=True), required=True)
def test_shapes(config):
    """Local sanity check to ensure the tensor flow works correctly"""
    cfg = OmegaConf.load(config)
    click.echo(f"🧪 Testing shapes with Latent Dim: {cfg.model.latent_dim}")
    
    vision = Vision(out_dims=cfg.model.latent_dim)
    memory = Memory(latent_dim=cfg.model.latent_dim)
    decoder = VisualDecoder(input_latent_dim=cfg.model.latent_dim)
    
    # Simulate batch [B, S, C, H, W]
    dummy_in = torch.randn(1, 10, 3, 224, 224)
    
    with torch.no_grad():
        z = vision(dummy_in)
        click.echo(f"Vision Output: {z.shape}")
        
        z_pred = memory(z[:, :-1, :]) # Teacher forcing input
        click.echo(f"Memory Prediction: {z_pred.shape}")
        
        video_out = decoder(z_pred)
        click.echo(f"Decoder Reconstruction: {video_out.shape}")
    
    click.echo("✅ Pipeline shapes are valid.")

if __name__ == "__main__":
    main()
