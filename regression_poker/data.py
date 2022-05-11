import numpy as np
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from regression_poker import config as cfg

class HandDataset(Dataset):
    def __init__(self, memory):
        self.memory = memory

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        X, y = self.memory[idx][0]
        return X, int(y)


class HandDataModule(pl.LightningDataModule):
    def __init__(self, memory):
        super().__init__()
        self.memory = memory


    def setup(self, stage):
        self.mnist_train = HandDataset(self.memory)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=cfg.DQNOptions.batch_size)
