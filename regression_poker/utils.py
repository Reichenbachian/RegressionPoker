from collections import deque, namedtuple
import itertools

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

number_conversion = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
ALL_CARDS = [x+y for x, y in itertools.product(["C", "D", "H", "S"], number_conversion)]
conversion_deck = dict(zip(ALL_CARDS, range(52)))

def convert_card_to_number(card):
    return conversion_deck[card]


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, *args):
        """Save a transition"""
        self.memory.append(args)
        if len(self.memory) > self.capacity:
            self.memory = self.memory[1:]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __getitem__(self, idx):
        return self.memory[idx]

    def __len__(self):
        return len(self.memory)
