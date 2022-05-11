import click
from pypokerengine.api.game import setup_config, start_poker
from regression_poker.agents.random import RandomPlayer
from regression_poker.agents.nn import NNPlayer
from regression_poker.agents.honest import HonestPlayer
from regression_poker import config as cfg

@click.command()
def main():
    nn1 = NNPlayer()
    nn2 = NNPlayer()
    while True:
        config = setup_config(max_round=cfg.max_round,
                              initial_stack=cfg.initial_stack,
                              small_blind_amount=cfg.small_blind_amount)
        config.register_player(name="p1", algorithm=nn1)
        config.register_player(name="p1", algorithm=nn2)
        # config.register_player(name="p2", algorithm=HonestPlayer())
        game_result = start_poker(config, verbose=1)

if __name__ == "__main__":
    main()