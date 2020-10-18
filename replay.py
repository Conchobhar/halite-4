# From https://www.kaggle.com/solverworld/replay-your-agent-on-a-specific-step
import json, time, os
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.utils import structify
from utils.base import write_html

from agent.main import agent  # CONFIG - MANUALLY IMPORT BOT
# from bots.v40f import agent  # CONFIG - MANUALLY IMPORT BOT
# CONFIG - REPLAY AND CORRESPONDING ID
pathdl = Path(f'{os.environ["HOME"]}/Downloads/')
paths = pathdl.glob('*.json')
path_replay = max(paths, key=lambda p:p.stat().st_ctime)


def replay_match(path, step=0):
    """Replay game to a specific step - necessary for recreating stateful values."""
    with open(path,encoding='ascii') as f:
        match = json.load(f)
    env = make("halite", configuration=match['configuration'], steps=match['steps'])
    myid = [pid for pid, name in enumerate(match['info']['TeamNames']) if name == "Ready Salted"][0]
    config = env.configuration
    # env already done - can write out full replay
    t = env.render(mode='html', return_obj=True, width=800, height=800, header=False, controls=True)
    write_html(t, 'replay.html')
    # If agent carries state across turns, need to run through all steps, or can directly index into a step otherwise
    # check that we are correct player
    # print('My Id: ', board.current_player_id, board.current_player)
    print(f'Running for:\n\t{path}\n\t{agent.__module__}\n\tID = {myid}\n')
    actTime = 0
    for step in range(400):
        state = match['steps'][step][0]  # list of length 1 for each step
        obs = state['observation']  # these are observations at this step
        obs['player'] = myid  # change the player to the one we want to inspect
        obs = structify(obs)  # turn the dict's into structures with attributes
        icon = '\|/-'[(obs.step+1) % 4]
        t0 = time.time()
        ret = agent(obs, config)
        actTime = time.time() - t0
        print(f'{icon} step+1: {obs.step +1} StepTime:{round(actTime,2)}')#, end="\r", flush=True)


replay_match(path_replay)
