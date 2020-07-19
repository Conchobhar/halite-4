from kaggle_environments.envs.halite.helpers import Board, ShipAction, ShipyardAction, Ship
from collections import namedtuple, OrderedDict
import numpy as np
np.seterr(all='raise')
"""NEXT
    -test and debug v0
    -build yard when there are zero
    -ship spawn kill point
    
    -each ship should have its own threat modifier for cells
        -ve if ship.halite < avg. enemy
        +ve if ship.halite > avg. enemy
    -while mining, low hlt ships should try to wall perimeter
    -build extra yard at some point
"""

"""Items for ship log
    'role'       What is ships purpose?
    'spot'       Where is ship focused on?
    'p_action'   potential action
    'p_point'    potential resultant point (from action)
    'set_action' Confirmed move action
    'set_point'  Confirmed resultant point (from action)
    """
# logEntry = namedtuple(
#     'logEntry',
#     ['role', 'spot', 'p_action', 'p_point', 'set_action', 'set_point'])

# Map move action to positional delta
adelta = {
    ShipAction.NORTH: (0, 1),
    ShipAction.EAST: (1, 0),
    ShipAction.SOUTH: (0, -1),
    ShipAction.WEST: (-1, 0),
}


class LogEntry:
    """Like a dictionary, without the ['fluff']"""
    def __init__(self):
        self.role = 'NEW'
        self.spot = 'NEW'
        self.yard = None
        self.p_action = None
        self.p_point = None
        self.set_action = None
        self.set_point = None


class Log(dict):
    """Keys are ships. Values are a LogEntry()"""
    def __init__(self):
        super().__init__()

    @property
    def spots(self):  # Assigned harvest spots
        return [x.spot for x in self.values() if x.spot is not None]

    @property
    def set_points(self):  # Assigned
        return [x.set_points for x in self.values() if x.spot is not None]


LOG = Log()

C = {
    'halite_harvest_minimum': 75,  # todo - should be function of map avg. and local avg.
    'ship_carry_maximum': 300,
}


class MyAgent:

    def __init__(self, config):
        self.board = None
        self.board_prev = None
        self.me = None
        self.dim = config.size
        self.mid = config.size // 2
        self.harvest_spot_values = None
        self.enemy_ship_point = None
        self.mean_halite = None
        self.yardcount = None
        self.prospective_yard_pos = None

    def update(self):
        pass

    def refresh_ships(self):
        """Update ship tasks"""
        ids = list(LOG.keys())
        for s in self.me.ships:
            if s in LOG:  # reconnect log
                s.log = LOG[s]
                ids.remove(s)
            else:  # mark new ships - update global LOG and add attr for ships entry
                LOG[s] = LogEntry()
                s.log = LOG[s]
        for s in ids:  # remaining ship ids therefore have crashed
            del LOG[s]

    def dist(self, p1, p2):
        p1, p2 = np.array(p1), np.array(p2)
        p = abs(p1-p2)
        y, x = min(p[0], self.dim - p[0]), min(p[1], self.dim - p[1])
        return abs(sum([x, y]))

    def get_nearest_shipyard(self, pos):
        if self.yardcount == 0:
            return self.prospective_yard_pos
        if self.yardcount == 1:
            return [sy for sy in self.board.shipyards.values()][0]
        for sy in self.board.shipyards:
            raise BaseException('NOT IMPLEMENTED YET')
        shipyard = ...
        return shipyard

    def generate_harvest_values(self):
        if self.yardcount == 1:
            d = OrderedDict()
            pos_yards = [x.cell.position for x in self.board.shipyards.values()]
            for pos, cell in self.board.cells.items():
                if pos not in pos_yards:
                    sy_pos = self.get_nearest_shipyard(pos).position
                    d[pos] = cell.halite / (self.dist(pos, sy_pos)**0.5)
            self.harvest_spot_values = sorted(d.items(), key=lambda x: -x[1])
            # todo - weight by threat == smoothed ship density, shipThreat = "avg ship halite"/shipHalite

    def assign_role(self, ship):
        ship.log.role = 'HARVESTER'
        ship.log.spot = None

    def determine_best_harvest_spot(self, ship):
        # Choose a spot to harvest
        spots_with_min_halite = [(spot, value) for spot, value in self.harvest_spot_values
            if self.board.cells[spot].halite > C['halite_harvest_minimum']]
        for spot, value in spots_with_min_halite:
            if spot not in LOG.spots:
                return spot
        raise BaseException('No spot found - need to implement aggr contingency at this point')

    def map_cyclic_coords(self, x):
        """Map higher half of coordinate space to its -ve equivalent
        e.g. for board dimension of length 5:
            (0,1,2,3,4,5) --> (0,1,2,-2,-1,0)"""
        return (x + self.mid) % 5 - self.mid

    def is_pos_occupied_by_threat(self, ship, ppos, assume_harvest=True):
        cell = self.board.cells[ppos]
        ppos_ship = cell.ship
        if ppos_ship is not None:
            is_occupied_by_threat = (
                    ppos_ship.player_id != self.me.id and
                    (ppos_ship.halite + cell.halite*self.board.configuration.collect_rate) < ship.halite)
        else:
            is_occupied_by_threat = False
        return is_occupied_by_threat

    def move_to_target(self, ship, pt):
        """Normalize coordinates and determine best action for approaching target.
        ps - pos of ship
        pt - pos of target
        Normalize:  translate origin to ps (i.e. subtract ps from pt)
                    map higher half coords to -ve values
        Avoid actions that are potentially unsafe

        Ideally would rate every option based on threats vs potential to win encounter.
        """
        #TODO maybe - prioritize based on ship threat density
        ps = ship.position
        pnorm = (self.map_cyclic_coords(pt[0] - ps[0]),
                 self.map_cyclic_coords(pt[1] - ps[1]))
        actions = {
            ShipAction.NORTH: (1 if pnorm[1] > 0 else -1),
            ShipAction.EAST:  (1 if pnorm[0] > 0 else -1),
            ShipAction.SOUTH: (1 if pnorm[1] < 0 else -1),
            ShipAction.WEST:  (1 if pnorm[0] < 0 else -1),
            None: 0,
        }
        chosen_action = 'UNDECIDED'
        n_conditions = 3
        while chosen_action is 'UNDECIDED':
            # for each possible action, in order of preference, determine if safe
            # If no action is safe, reduce the amount safety conditions until no options are left.
            for action in sorted(actions, key=actions.get)[::-1]:
                ppos = ps.translate(adelta[action], self.dim)
                ppos_adjs = [ppos.translate(adelta[a], self.dim) for a in actions if a is not None]
                ppos_adjs.remove((ps.x, ps.y))
                # not occupied by enemy ship with less halite
                is_not_occupied_by_threat = not self.is_pos_occupied_by_threat(ship, ppos, assume_harvest=True)
                is_not_occupied_by_self = (ppos not in LOG.set_points)
                is_not_occupied_by_potential_threats = any([self.is_pos_occupied_by_threat(ship, ppos_adj) for ppos_adj in ppos_adjs])
                # Conditions are ordered by priority
                conditions = [is_not_occupied_by_threat, is_not_occupied_by_self, is_not_occupied_by_potential_threats]
                if all(conditions[0:n_conditions-1]):
                    chosen_action = action
                else:
                    n_conditions -= 1
                if n_conditions == 0:
                    chosen_action = None  # No good moves found TODO log this
        return chosen_action

    def determine_best_harvest_action(self, ship):
        if not ship.position == ship.log.spot:  # move to spot
            ship.log.p_action = self.move_to_target(ship, ship.log.spot)
            ship.log.p_point = ship.position + ship.log.p_action
        else: # harvest
            ship.log.p_action = None
            ship.log.p_point = ship.position

    def determine_best_deposit_action(self, ship):
        if not ship.position == ship.log.yard:  # move to spot
            ship.log.p_action = self.move_to_target(ship.position, ship.log.yard)
            ship.log.p_point = ship.position + ship.log.p_action
        else:
            raise BaseException('depositor but ship is on yard pos and didnt switch role?')

    def determine_ship_action(self, ship):
        """TODO possible issues with role switch conditions here."""
        if ship.log.role == 'NEW':
            self.assign_role(ship)
        ship.log.yard = self.get_nearest_shipyard(ship.position)
        if ship.log.role == 'DESPOITOR' and ship.position == ship.log.yard:
            ship.log.role = 'HARVESTER'
        if ship.log.role == 'HARVESTER':
            if ship.log.spot is None or self.board.cells[ship.log.spot].halite > C['halite_harvest_minimum']:
                ship.log.spot = self.determine_best_harvest_spot(ship)
            if ship.halite > C['ship_carry_maximum']:
                ship.log.role = 'DEPOSITOR'
            else:
                self.determine_best_harvest_action(ship)
        if ship.log.role == 'DEPOSITOR':
            self.determine_best_deposit_action(ship)

    def get_actions(self, obs, config):
        self.board = Board(obs, config)
        self.me = self.board.current_player
        me, self.b, self.bp = self.me, self.board, self.board_prev  # just for shorthand
        self.refresh_ships()
        spawncount = 0
        self.yardcount = len(self.me.shipyards)
        self.mean_halite = int(np.mean([cell.halite for cell in self.board.cells.values() if cell.halite != 0]))
        C['halite_harvest_minimum'] = self.mean_halite
        C['ship_carry_maximum'] = self.mean_halite * 4
        self.enemy_ship_points = [ship.position for plr in self.board.players.values()
                               if plr is not self.me for ship in plr.ships]
        self.generate_harvest_values()
        while len(me.next_actions) != len(me.ships):
            # Main ship loop - iterate until each ship has an action
            for ship in me.ships:
                # TODO - order ships by priority. should be done after yard building assigned
                # TODO - ships on SY should go first
                # If first turn, create yard
                # TODO - if ndocks < 1, prioritize building yard
                if obs.step == 0:
                    ship.next_action = ShipAction.CONVERT
                    self.prospective_yard_pos = ship.position
                else:
                    # ship.next_action = ShipAction.NORTH
                    self.determine_ship_action(ship)

        # Ship building
        for shipyard in me.shipyards:
            # If we can afford spawn, considering cumulation of other SY spawns.
            if (me.halite - spawncount * config.spawnCost) >= config.spawnCost:  # TODO - if haveHalite and stillBuilding and noShipMovingHere
                shipyard.next_action = ShipyardAction.SPAWN
                spawncount += 1
        self.board_prev = self.board
        return me.next_actions



def agent(obs, config):
    global myBot
    if 'myBot' not in globals():
        myBot = {obs.player: MyAgent(config)}
    # myBot[obs.player].update(obs, config)
    actions = myBot[obs.player].get_actions(obs, config)
    return actions


# def render()...
#         html = f'<iframe srcdoc="{player_html}" width="{width}" height="{height}" frameborder="0"></iframe> '
#         if "return_obj" in kwargs: return html
#         display(HTML(html))
#     elif mode == "json":


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
