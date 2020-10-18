from kaggle_environments.envs.halite.helpers import *

global myOtherGlobal
globe = {}


class MyAgent:

    def __init__(self, obs, config):
        self.board = Board(obs, config)
        self.config = self.board.configuration
        self.me = self.board.current_player
        self.dim = config.size

    def get_actions(self):
        ...

def agent(obs, config):
    global myBot
    if 'myBot' not in globals():
        myBot = {obs.player: MyAgent(obs, config)}
    actions = myBot[obs.player].get_actions(obs, config)
    return actions
