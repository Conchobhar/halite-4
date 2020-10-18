from kaggle_environments import make
from time import time
from utils.base import write_html
from agent.main import agent

"""SEEDS
"""
config = {"size": 21, "startingHalite": 24000, "randomSeed": 131300848}
env = make("halite", configuration=config, debug=True)
trainer = env.train([None, 'random', 'random', 'random'])
trainer = env.train([None, 'bots/v40e.py', 'bots/v40e.py', 'agent/main.py'])


def play():
    """Play bots locally."""
    obs = trainer.reset()
    actTime = 0
    while not env.done:
        t0 = time()
        my_actions = agent(obs, env.configuration)
        actTime = time() - t0
        icon = ['>---', '->--', '-->-', '--->',][(obs.step+1) % 4]
        print(f'{icon} step+1: {obs.step +1} n actions: {len(my_actions)} prevActTime:{actTime}', end="\r", flush=True)
        obs, reward, done, info = trainer.step(my_actions)
        t = env.render(mode='html', return_obj=True, width=800, height=800, header=False, controls=True)
        write_html(t, 'render.html')


t0 = time()
play()
print(time() - t0)
