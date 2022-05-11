'''
A simple agent that looks at current hand and estimates win percentage
then bets. Can't raise.
'''
import numpy as np
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F

from regression_poker.utils import convert_card_to_number


class LinearCall(pl.LightningModule):
    '''
    We are going to outpu the odds on us winning the hand

    0: We lose hand (Fold)
    1: We win hand (Call)
    '''
    VEC_SIZE = 52 * 5
    def __init__(self, n_layers=0, hidden_size=128):
        super().__init__()
        assert n_layers == 0, "Haven't gotten around to this."
        self.embd_layer = nn.Linear(self.VEC_SIZE, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 2)
        self.loss = nn.CrossEntropyLoss()

    def convert_to_input_vec(self, valid_actions, hole_card, round_state):
        community_cards = round_state['community_card']
        card_numbers = [convert_card_to_number(x) for x in hole_card + community_cards]
        arr = np.zeros(self.VEC_SIZE, dtype=np.float32)
        for card in card_numbers:
            arr[card] = 1

        return torch.from_numpy(arr)

    def convert_to_output(self, y):
        action = y.argmax().item()
        if action == 0:
            return "fold", 0
        return "call", None

    def forward(self, x):
        x = self.embd_layer(x)
        x = F.relu(x)
        x = self.output_layer(x)
        # x = F.softmax(x, dim=-1) # Don't do it
        return x

    def train_dataloader(self):
        return DataLoader(self.train_ds)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        return loss
        # self.log('train_loss', loss)
        # return loss

    # def validation_step(self, val_batch, batch_idx):
    #     x, y = val_batch
    #     x = x.view(x.size(0), -1)
    #     z = self.encoder(x)
    #     x_hat = self.decoder(z)
    #     loss = F.mse_loss(x_hat, x)
    #     self.log('val_loss', loss)
