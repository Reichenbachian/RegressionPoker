# RegressionPoker

**Paper Summary:** Previous poker bots often struggle with the massive state space of the game. We simplify the state space by treating the problem as a regression problem, instead of independent bets. By treating it as a regression task, we vastly simplify the state space and problem as a whole. Using this new formulation of the problem, we train a poker bot using self-play and experiential replay in the OpenAI Gym environment.

**Repo Summary:** Upon running the `train.py` script, two neural-network based poker bots will play Heads-Up no-limit poker continually. After `train_every=1000` hands, they will run a brief training session. Once either runs out of money, the states reset and the training continues.

**Run Instructions:**
First, create a virtual environment using `virtualenv venv -p $(which python3)`.
Then, we need to activate it - `. venv/bin/activate`
Install our requirements - `pip install -r requirements.txt`.
Then we can run a training session of our self-play neural networks. Note that we need to change the PYTHONPATH to include our regression_poker home directory.
```
PYTHONPATH=/home/localhost/Desktop/courses/computation_intelligence_for_games/regression_poker/:$PYTHONPATH python scripts/train.py
```