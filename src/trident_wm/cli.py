import click
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from trident_wm.vision import Vision
from trident_wm.memory import Memory
from trident_wm.controller import VisualDecoder
from trident_wm.system import WorldSystem
from trident_wm.datamodule import PushTDataModule

@click.group()
def main():
    """🔱 trident-WM: World Model Sprint (V-M-D)"""
    pass

@main.command()
@click.option('--data', default="lerobot/pusht", help='HF Repo ID or local path')
@click.option('--epochs', default=10, help='Total training epochs')
@click.option('--batch_size', default=8, help='Batch size for CPU/GPU')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--device', default='cpu', type=click.Choice(['cpu', 'gpu']), help='Compute device')
def train(data, epochs, batch_size, lr, device):
    """Train the Vision Neck, Memory Transformer, and Visual Decoder"""
    click.echo(f"🧬 Initializing V-M-D Pillars...")
    
    # Initialize Pillars
    vision = Vision()
    memory = Memory()
    decoder = VisualDecoder()
    
    # Initialize System
    system = WorldSystem(vision, memory, decoder, lr=lr)
    
    # Initialize DataModule
    dm = PushTDataModule(repo_id=data, batch_size=batch_size)
    
    # Setup Logger
    logger = WandbLogger(project="trident-wm", name="V-M-D-Training")
    
    # Initialize Trainer
    trainer = L.Trainer(
        accelerator=device,
        devices=1,
        max_epochs=epochs,
        logger=logger,
        precision="16-mixed" if device == 'gpu' else 32,
        log_every_n_steps=10
    )
    
    click.echo(f"🚀 Starting training on {device}...")
    trainer.fit(system, datamodule=dm)

@main.command()
@click.option('--checkpoint', required=True, help='Path to the .ckpt file')
@click.option('--data', default="lerobot/pusht", help='HF Repo ID or local path')
@click.option('--batch_size', default=4)
def evaluate(checkpoint, data, batch_size):
    """Evaluate World Model imagination on the unseen test split"""
    click.echo(f"🌙 Loading System from checkpoint: {checkpoint}")
    
    # Initialize Pillars (required for loading state_dict)
    vision = Vision()
    memory = Memory()
    decoder = VisualDecoder()
    
    # Load System
    system = WorldSystem.load_from_checkpoint(
        checkpoint, 
        vision=vision, 
        memory=memory, 
        decoder=decoder
    )
    
    # Initialize DataModule
    dm = PushTDataModule(repo_id=data, batch_size=batch_size)
    
    # Setup Logger for Evaluation results
    logger = WandbLogger(project="trident-wm", name="V-M-D-Evaluation")
    
    # Initialize Trainer for Testing
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=logger
    )
    
    click.echo(f"🧪 Running Test Split Evaluation...")
    trainer.test(system, datamodule=dm)

@main.command()
def test_shapes():
    """Quick sanity check for tensor shapes across the pipeline"""
    click.echo("🧪 Testing Pillar Shape Consistency...")
    
    v = Vision()
    m = Memory()
    d = VisualDecoder()
    
    # Batch=1, Seq=10, C=3, H=224, W=224
    dummy_in = torch.randn(1, 10, 3, 224, 224)
    
    with torch.no_grad():
        z = v(dummy_in)
        click.echo(f"Vision Output: {z.shape} (Expected: [1, 10, 256])")
        
        z_hat = m(z[:, :-1, :])
        click.echo(f"Memory Output: {z_hat.shape} (Expected: [1, 9, 256])")
        
        video_hat = d(z_hat)
        click.echo(f"Decoder Output: {video_hat.shape} (Expected: [1, 9, 3, 224, 224])")
    
    click.echo("✅ All shapes are consistent.")

if __name__ == "__main__":
    main()
