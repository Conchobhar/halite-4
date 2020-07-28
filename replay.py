# From https://www.kaggle.com/solverworld/replay-your-agent-on-a-specific-step
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from kaggle_environments.utils import structify
import argparse
import pprint
import json
from pathlib import Path

from utils.base import write_html
from submissions.v2 import agent

path_replay = Path('./replays/v2-1905893.json')


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
    # Need to run my agent to develop stateful values
    # check that we are correct player
    # print('My Id: ', board.current_player_id, board.current_player)
    print(f'Running for: {agent.__module__}')
    for step in range(400):
        state = match['steps'][step][0]  # list of length 1 for each step
        obs = state['observation']  # these are observations at this step
        obs['player'] = playerid  # change the player to the one we want to inspect
        board = Board(obs, config)
        obs = structify(obs)  # turn the dict's into structures with attributes
        # This is our agent recreating what happened on this step
        # ret=sub.agent(obs, config)
        ret = agent(obs, config)
        print(f'step: {obs.step}')



'''
replay_match(options.file, options.step, options.id)
'''
replay_match(path_replay, 0)
