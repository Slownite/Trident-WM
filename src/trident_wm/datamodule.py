import lightning as L
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from trident_wm.constants import SEQ_LEN

class PushTDataModule(L.LightningDataModule):
    def __init__(
        self, 
        repo_id: str = "lerobot/pusht", 
        batch_size: int = 32, 
        num_workers: int = 4
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = None
        
        # DINOv2 Requirement: Dimensions must be multiples of 14 (e.g., 224x224)
        # We use v2 transforms for efficiency
        self.image_transforms = v2.Compose([
            v2.Resize((224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ])
        
        self.delta_timestamps = {
            "observation.image": [i * 0.1 for i in range(SEQ_LEN)],
            "observation.state": [0.0]
        }

    def prepare_data(self):
        LeRobotDataset(self.hparams.repo_id)

    def setup(self, stage: str | None = None):
        if self.dataset is None:
            # Pass image_transforms directly to LeRobotDataset
            self.dataset = LeRobotDataset(
                repo_id=self.hparams.repo_id,
                delta_timestamps=self.delta_timestamps,
                image_transforms=self.image_transforms
            )
            
            total_size = len(self.dataset)
            train_size = int(0.8 * total_size)
            val_size = int(0.1 * total_size)
            
            self.train_ds = Subset(self.dataset, range(0, train_size))
            self.val_ds = Subset(self.dataset, range(train_size, train_size + val_size))
            self.test_ds = Subset(self.dataset, range(train_size + val_size, total_size))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False
        )
