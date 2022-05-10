from pypokerengine.players import BasePokerPlayer
import random

class RandomPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
         # If we can't raise, don't
        if valid_actions[2]['amount']['min'] == -1:
            del valid_actions[2]

        # Choose random action
        action = random.choice(valid_actions)
        if action['action'] == 'raise':
            amount = random.randrange(action['amount']['min'], action['amount']['max'])
        else:
            amount = action["amount"]
        return action['action'], amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
