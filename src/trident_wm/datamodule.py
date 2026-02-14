import lightning as L
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from trident_wm.constants import SEQ_LEN

class PushTDataModule(L.LightningDataModule):
    """
    DataModule for the Push-T robotics task using LeRobot datasets.
    Handles downloading, temporal sequence chunking, and GPU batching.
    """
    
    def __init__(
        self, 
        repo_id: str = "lerobot/pusht", 
        batch_size: int = 32, 
        num_workers: int = 4
    ):
        super().__init__()
        # 1. Initialization: Capture parameters for reproducibility
        self.save_hyperparameters()
        self.dataset = None

    def prepare_data(self):
        """
        Step 1: Preparation (Single-process)
        Download the dataset to the local cache. Lightning ensures this 
        runs once even in multi-GPU environments.
        """
        LeRobotDataset(self.hparams.repo_id)

    def setup(self, stage: str | None = None):
        """
        Step 2: Setup (Every-process)
        Uses the new dictionary-based delta_timestamps API to create 
        temporal stacks for Vision and Memory pillars.
        """
        if self.dataset is None:
            # We define a window of SEQ_LEN steps at 10Hz (0.1s intervals)
            # For a World Model, we typically want current and future frames
            timestamps = [i * 0.1 for i in range(SEQ_LEN)]

            # New API requirement: Map the window to specific dataset keys
            delta_timestamps = {
                "observation.image": timestamps,
                "action": timestamps,
                "observation.state": [0.0] # Current state only
            }

            self.dataset = LeRobotDataset(
                repo_id=self.hparams.repo_id,
                delta_timestamps=delta_timestamps
            )

    def train_dataloader(self):
        """
        Step 3: Training Stream
        Returns a DataLoader optimized for development or deploy on cloud.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,  # Speeds up tensor transfer to GPU
            drop_last=True     # Prevents partial batches from crashing Triton kernels
        )

    def val_dataloader(self):
        """
        Step 4: Validation Stream
        """
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False
        )
