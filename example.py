import os
from pathlib import Path
from kaggle_environments import evaluate, make
from agent.base import agent
from time import time

pathroot = Path('/home/xu/work/kaggle/halite/')
config = {"size": 7, "startingHalite": 1000, "randomSeed":0}
env = make("halite", configuration=config, debug=True)
trainer = env.train([None, 'random'])
"""observation
'player', 'step', 'halite',         'players'
 0         1       MAPSIZE**2 array  ...
 
'players'[0]
your halite, {}, {'1-1': [122,0]}

"""


def write_html(html):
    with open(os.path.join(pathroot, 'html/render.html'), 'w') as f:
        f.write(html)


def play():
    obs = trainer.reset()
    while not env.done:
        my_actions = agent(obs, env.configuration)
        print(f"Step: {obs['step']}, n actions {len(my_actions)}")
        obs, reward, done, info = trainer.step(my_actions)
        t = env.render(mode='html', return_obj=True, width=800, height=800, header=False, controls=True)
        write_html(t)


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
