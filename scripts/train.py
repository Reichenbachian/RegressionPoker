import click
from pypokerengine.api.game import setup_config, start_poker
from regression_poker.agents.random import RandomPlayer
from regression_poker import config as cfg

@click.command()
def main():
    config = setup_config(max_round=cfg.max_round,
                          initial_stack=cfg.initial_stack,
                          small_blind_amount=cfg.small_blind_amount)
    config.register_player(name="p1", algorithm=RandomPlayer())
    config.register_player(name="p2", algorithm=RandomPlayer())
    game_result = start_poker(config, verbose=1)

if __name__ == "__main__":
    main()