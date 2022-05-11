from pypokerengine.players import BasePokerPlayer
import pytorch_lightning as pl
import random
import torch
import torch.nn as nn
from tqdm import tqdm

from regression_poker.config import DQNOptions as dqn_cfg
from regression_poker.agents.linear_call import LinearCall
from regression_poker.utils import ReplayMemory
from regression_poker.data import HandDataModule


class NNPlayer(BasePokerPlayer):
    agents = {'linear_call': LinearCall}
    def __init__(self, agent_name='linear_call'):
        super().__init__()
        assert agent_name in self.agents, "Agent doesn't exist"
        self.net = self.agents[agent_name]()
        self.memory = ReplayMemory(dqn_cfg.mem_size)
        self.states = []
        self.hand_count = 0
        self.pbar = tqdm(total=dqn_cfg.train_every)

    def train(self):
        '''
        Run a training session
        '''
        ds = HandDataModule(self.memory)
        print("Training...")
        trainer = pl.Trainer(gpus=1, precision=16, limit_train_batches=0.5)
        trainer.fit(self.net, ds) #, val_loader)

    def declare_action(self, valid_actions, hole_card, round_state):
        '''
        We're just inferring through the network on this
        '''
        if random.random() < dqn_cfg.random_pctg:
            action = random.choice(valid_actions)
            if action['action'] == 'raise':
                min, max = action['amount']['min'], action['amount']['max']
                return action['action'], (min + max) // 2
            return action['action'], action['amount']

        with torch.no_grad():
            x = self.net.convert_to_input_vec(valid_actions, hole_card, round_state)
            y = self.net(x)
            action, amount = self.net.convert_to_output(y)
            
            # Put in correct call amount
            if action == "call":
                amount = valid_actions[1]['amount']
            self.states.append(x)

        return action, amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        is_winner = self.uuid in [x['uuid'] for x in winners]
        for state in self.states:
            self.memory.push((state, is_winner))
        self.states = []

        if (self.hand_count % dqn_cfg.train_every == 0) and \
            (self.hand_count > 0):
            self.train()
            self.pbar = tqdm(total=dqn_cfg.train_every)
        self.hand_count += 1
        self.pbar.update()
