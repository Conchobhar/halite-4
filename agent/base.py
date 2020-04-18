def my_agent(obs):
    actions = {}  # dict of {shipid:actions}
    if obs.step == 1:
        ship_id = list(obs.players[obs.player][2].keys())[0]
        actions[ship_id] = 'CONVERT'
        return actions
    ship_id = list(obs.players[obs.player][2].keys())[0]
    ship_action = 'NORTH'
    if ship_action is not None:
        actions[ship_id] = ship_action
    return actions
