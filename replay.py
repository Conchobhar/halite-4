# From https://www.kaggle.com/solverworld/replay-your-agent-on-a-specific-step
import json,sys, time
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from kaggle_environments.utils import structify

from utils.base import write_html

try:
    from bots import latest
    agent = latest()
except:
    from bots.v11 import agent  # CONFIG - MANUALLY IMPORT BOT
myid = 1

# CONFIG - REPLAY AND CORRESPONDING ID
pathdl = Path('/home/xu/Downloads/')
# path_replay = pathdl / '1984103.json'
paths = pathdl.glob('*.json')
path_replay = max(paths, key=lambda p:p.stat().st_ctime)


'''
Load a json eposode record and play a particular step with a given agent
as a given player.  Use to determine failure or debug actions on a particular step
of a previously played episode
'''
'''
Use these lines to make a command line processor:

parser = argparse.ArgumentParser(
    description="replay a json file",
    epilog='''
''')
parser.add_argument("file", type=str, help="json file to play",nargs='?',default='1233550.json')
parser.add_argument("--step", type=int, help="step to play",default=0)
parser.add_argument("--id", type=int, help="player to be",default=0)
options = parser.parse_args()
'''


def replay_match(path, playerid, step=0):
    with open(path, 'r') as f:
        match = json.load(f)
    env = make("halite", configuration=match['configuration'], steps=match['steps'])
    config = env.configuration
    # env already done - can write out full replay
    t = env.render(mode='html', return_obj=True, width=800, height=800, header=False, controls=True)
    write_html(t, 'replay.html')
    # If agent carries state across turns, need to run through all steps, or can directly index into a step otherwise
    # check that we are correct player
    # print('My Id: ', board.current_player_id, board.current_player)
    print(f'Running for:\n\t{path}\n\t{agent.__module__}')
    for step in range(400):
        state = match['steps'][step][0]  # list of length 1 for each step
        obs = state['observation']  # these are observations at this step
        obs['player'] = playerid  # change the player to the one we want to inspect
        board = Board(obs, config)
        obs = structify(obs)  # turn the dict's into structures with attributes
        icon = '+x'[(obs.step+1) % 2]
        print(f'{icon} step+1: {obs.step +1}', end="\r", flush=True)
        # sys.stdout.write(f'step+1: {obs.step +1}')
        # sys.stdout.flush()
        # restart_line()
        ret = agent(obs, config)


replay_match(path_replay, myid)
