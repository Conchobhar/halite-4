import os
from pathlib import Path
from kaggle_environments import evaluate, make
from agent.base import agent

pathroot = Path('/home/xu/work/kaggle/halite/')
config = {"size": 7, "startingHalite": 1000}
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
        print(f"Step: {obs['step']} actions {my_actions}")
        obs, reward, done, info = trainer.step(my_actions)
        t = env.render(mode='html', return_obj=True, width=800, height=800, header=False, controls=True)
        write_html(t)


play()
t = env.render(mode='html', return_obj=True, width=800, height=800, header=False, controls=True)
write_html(t)

# Use env to Play:
# env.run(["./submission.py", "random"])
# env.render(mode="ipython", width=800, height=600)
