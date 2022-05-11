num_players = 2 # We are only playing heads up poker
max_round = 2000
initial_stack = 2000
small_blind_amount = 5

class DQNOptions:
	mem_size = 10000
	train_every = 1000
	# train_steps = 100
	batch_size = 16 #128
	random_pctg = 0.1
