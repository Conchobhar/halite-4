import os
from pathlib import Path
from kaggle_environments import evaluate, make
from agent.base import agent
from time import time
from utils.base import write_html
from bots import v1, home2p1

"""SEEDS
1984826053 - t15 ships deposits then returns to same spot
1695788596 - t5 gridflock - can be fixed with  pathing"""
config = {"size": 21, "startingHalite": 24000, "randomSeed": 824666594}
env = make("halite", configuration=config, debug=True)
trainer = env.train([None, 'bots/home2p1.py', 'random', 'random'])
# config = {"size": 21, "startingHalite": 24000, "randomSeed": 824666594}
# env = make("halite", configuration=config, debug=True)
# trainer = env.train([None, 'random',])


def play():
    obs = trainer.reset()
    while not env.done:
        my_actions = agent(obs, env.configuration)
        print(f"Step+1: {obs['step']+1}, n actions {len(my_actions)}")
        obs, reward, done, info = trainer.step(my_actions)
        t = env.render(mode='html', return_obj=True, width=800, height=800, header=False, controls=True)
        write_html(t, 'render.html')


t0 = time()
play()
print(time() - t0)

# Use env to Play:
# env.run(["./submission.py", "random"])
# env.render(mode="ipython", width=800, height=600)



# CONFIG
# {'size': 5,
#  'startingHalite': 1000,
#  'episodeSteps': 400,
#  'agentExec': 'LOCAL',
#  'agentTimeout': 30,
#  'actTimeout': 6,
#  'runTimeout': 9600,
#  'spawnCost': 500,
#  'convertCost': 500,
#  'moveCost': 0,
#  'collectRate': 0.25,
#  'regenRate': 0.02,
#  'maxCellHalite': 500}
