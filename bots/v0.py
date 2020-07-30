from kaggle_environments.envs.halite.helpers import Board, ShipAction, ShipyardAction, Ship, Shipyard
from collections import Counter, OrderedDict
from functools import lru_cache
from itertools import combinations
import numpy as np

np.seterr(all='raise')
"""NEXT
    -test and debug v0

    - improve harvest logic
        if spot.halite < threshold, go to nearby spot

    -assault role, enabled when not worth mining


    -build extra yard at some point

    -each ship should have its own threat modifier for cells
        -ve if ship.halite < avg. enemy
        +ve if ship.halite > avg. enemy
"""

"""Items for ship log
    'role'       What is ships purpose?
    'spot'       Where is ship focused on?
    'p_action'   potential action
    'p_point'    potential resultant point (from action)
    'set_action' Confirmed move action
    'set_point'  Confirmed resultant point (from action)
    """
# Map move action to positional delta
adelta = {
    ShipAction.NORTH: (0, 1),
    ShipAction.EAST: (1, 0),
    ShipAction.SOUTH: (0, -1),
    ShipAction.WEST: (-1, 0),
    'WAIT': (0, 0),
}

ainverse = {
    ShipAction.NORTH: ShipAction.SOUTH,
    ShipAction.EAST: ShipAction.WEST,
    ShipAction.SOUTH: ShipAction.NORTH,
    ShipAction.WEST: ShipAction.EAST,
    'WAIT': 'WAIT',
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
        self.resetable_names = ['p_action', 'p_point', 'set_action', 'set_point', ]

    def reset_turn_values(self):
        """Reset values that don't carry across turns."""
        for name in self.resetable_names:
            setattr(self, name, None)


class Log(dict):
    """Keys are ships. Values are a LogEntry()"""

    def __init__(self):
        super().__init__()
        # Keep a reference to me - necessary to extract `next_actions`
        self.me = None

    @property
    def spots(self):  # Assigned harvest spots
        return [x.spot for x in self.values() if x.spot is not None]

    @property
    def p_points(self):  # Potential next turn positions
        return [x.p_point for x in self.values() if x.p_point is not None]

    @property
    def set_points(self):  # Assigned next turn positions
        return [x.set_point for x in self.values() if x.set_point is not None]

    @property
    def set_actions(self):  # Assigned next turn positions
        return [x.set_action for x in self.values() if x.set_action is not None]

    @property
    def p_point2ship(self):  # point 2 potential ships map - excluding ships already set
        p2s = {}
        for ship in [s for s in self if s.log.set_action is None]:
            if ship.log.p_point not in p2s:
                p2s[ship.log.p_point] = [ship]
            else:
                p2s[ship.log.p_point].append(ship)
        return p2s

    @property
    def unset_ships(self):  # Assigned next turn positions
        return [s for s, log in self.items() if log.set_point is None]


global LOG
LOG = Log()


class MyAgent:

    def __init__(self, obs, config):
        self.board = None
        self.board_prev = None
        self.me = None
        self.dim = config.size
        self.mid = config.size // 2
        self.harvest_spot_values = None
        self.enemy_ship_points = None
        self.mean_halite = None
        self.yardcount = None
        self.prospective_yard = None
        self.action_iter = None
        self.keep_spawning_tripswitch = True
        self.cell_halite_minimum = None
        self.ship_carry_maximum = None  # calculated at start of each loop
        self.generate_constants()

    def generate_constants(self):
        self.cell_halite_minimum = 0

    def update(self):
        pass

    def refresh_ships(self):
        """Attach `me` to LOG
        Refresh log keys (i.e. ship obj) connecting to new state through ship.id
        Attach global LOG as attribute for conveniance."""
        global LOG
        old_log = LOG
        LOG = Log()  # Clears any crashed ships - LOG is rebuilt below
        ids = {s.id: le for s, le in old_log.items()}
        LOG.me = self.me
        for s in self.me.ships:
            # Default .next_action is None, which implies waiting.
            # Use this default to indicate that we have not yet decided on an action.
            s.next_action = 'NOTSET'
            if s.id in ids:  # reconnect log
                LOG[s] = ids[s.id]
                s.log = LOG[s]
                s.log.reset_turn_values()
            else:  # log new ships
                LOG[s] = LogEntry()
                s.log = LOG[s]

    @lru_cache(maxsize=64)
    def dist(self, p1, p2):
        p1, p2 = np.array(p1), np.array(p2)
        p = abs(p1 - p2)
        y, x = min(p[0], self.dim - p[0]), min(p[1], self.dim - p[1])
        return abs(sum([x, y]))

    def get_nearest_shipyard(self, pos):
        if self.yardcount == 0:
            return self.prospective_yard
        if self.yardcount == 1:
            return [sy for sy in self.board.shipyards.values()][0]
        for sy in self.board.shipyards:
            raise BaseException('NOT IMPLEMENTED YET')
        shipyard = ...
        return shipyard

    def generate_harvest_values(self):
        if self.yardcount > 0:
            d = OrderedDict()
            pos_yards = [x.cell.position for x in self.board.shipyards.values()]
            for pos, cell in self.board.cells.items():
                if pos not in pos_yards:
                    sy_pos = self.get_nearest_shipyard(pos).position
                    d[pos] = cell.halite / (self.dist(pos, sy_pos) ** 0.5)
            self.harvest_spot_values = sorted(d.items(), key=lambda x: -x[1])
            # todo - weight by threat == smoothed ship density, shipThreat = "avg ship halite"/shipHalite

    @staticmethod
    def assign_role(ship):
        ship.log.role = 'HVST'
        ship.log.spot = None

    def determine_best_harvest_spot(self, ship):
        # Choose a spot to harvest - values already sorted desceding.
        spots_with_min_halite = [(spot, value) for spot, value in self.harvest_spot_values]
        for spot, value in spots_with_min_halite:
            if spot not in LOG.spots:
                return spot
        # TODO - roles - assault
        self.keep_spawning_tripswitch = False
        # Share spots in this case
        for spot, value in spots_with_min_halite:
            return spot

    def map_cyclic_coords(self, x):
        """Map higher half of coordinate space to its -ve equivalent
        e.g. for board dimension of length 5:
            (0,1,2,3,4,5) --> (0,1,2,-2,-1,0)"""
        return (x + self.mid) % self.dim - self.mid

    def is_pos_occupied_by_threat(self, ship, ppos):
        """Don't consider a ship with equal halite a threat unless depositing."""
        cell = self.board.cells[ppos]
        ppos_ship = cell.ship
        if cell.shipyard is not None and cell.shipyard.player_id != self.me.id:
            is_occupied_by_threat = True
        elif ppos_ship is not None:
            if ship.log.role == 'DEP':
                is_occupied_by_threat = (
                        ppos_ship.player_id != self.me.id and ppos_ship.halite <= ship.halite)
            else:
                is_occupied_by_threat = (
                        ppos_ship.player_id != self.me.id and ppos_ship.halite < ship.halite)
        else:
            is_occupied_by_threat = False
        return is_occupied_by_threat

    def move_to_target(self, ship, pt):
        """Normalize coordinates and determine best action for approaching target.
        ship - ship moving
        pt - pos of target
        ps - pos of ship
        Normalize:  translate origin to ps (i.e. subtract ps from pt)
                    map higher half coords to -ve values
        Avoid actions that are potentially unsafe

        Ideally would rate every option based on threats vs potential to win encounter.
        """
        # TODO maybe - prioritize based on ship threat density
        ps = ship.position
        pnorm = (self.map_cyclic_coords(pt[0] - ps[0]),
                 self.map_cyclic_coords(pt[1] - ps[1]))
        actions = {
            ShipAction.NORTH: (1 if pnorm[1] > 0 else -1),
            ShipAction.EAST: (1 if pnorm[0] > 0 else -1),
            ShipAction.SOUTH: (1 if pnorm[1] < 0 else -1),
            ShipAction.WEST: (1 if pnorm[0] < 0 else -1),
            'WAIT': (1 if pnorm == (0, 0) else 0),
        }
        chosen_action = 'UNDECIDED'
        n_conditions = 3
        best_to_worst_actions = sorted(actions, key=actions.get)[::-1]
        while chosen_action == 'UNDECIDED':
            # for each possible action, in order of preference, determine if safe
            # If no action is safe, reduce the amount safety conditions until no options are left
            # TODO - Should try to take favourable conflicts, all else being equal
            for action in best_to_worst_actions:
                ppos = ps.translate(adelta[action], self.dim)
                action_inverse = ainverse[action]
                ppos_adjs = [ppos.translate(adelta[a], self.dim) for a in actions if a not in (None, action_inverse)]
                # not occupied by enemy ship with less halite
                is_not_occupied_by_threat = not self.is_pos_occupied_by_threat(ship, ppos)
                is_not_occupied_by_self = (ppos not in LOG.set_points)
                is_not_occupied_by_potential_threats = all(
                    [not self.is_pos_occupied_by_threat(ship, ppos_adj) for ppos_adj in ppos_adjs])
                # Conditions are ordered by priority
                conditions = [is_not_occupied_by_threat, is_not_occupied_by_self, is_not_occupied_by_potential_threats]
                if all(conditions[0:n_conditions]):
                    chosen_action = action
                    break
                else:
                    n_conditions -= 1
                if n_conditions == 0:
                    chosen_action = ShipAction.CONVERT  # No good moves found
                    # TODO log this - Might need to enforce .next_action here and bypass action resolve cycle.
                    break
        return chosen_action

    def determine_best_harvest_action(self, ship):
        if not ship.position == ship.log.spot:
            ship.log.p_action = self.move_to_target(ship, ship.log.spot)
            if ship.log.p_action != ShipAction.CONVERT:  # Will only convert if there are no safe moves.
                ship.log.p_point = ship.position.translate(adelta[ship.log.p_action], self.dim)
            else:
                ship.log.set_action = ShipAction.CONVERT
                ship.next_action = ShipAction.CONVERT
        else:  # harvest - identical for now!
            ship.log.p_action = self.move_to_target(ship, ship.log.spot)
            ship.log.p_point = ship.position.translate(adelta[ship.log.p_action], self.dim)

    def determine_best_deposit_action(self, ship):
        # yard_position = ship.log.yard.position if ship.log.yard.position is not None else self.prospective_yard
        if not ship.position == ship.log.yard.position:  # move to spot
            ship.log.p_action = self.move_to_target(ship, ship.log.yard.position)
            ship.log.p_point = ship.position.translate(adelta[ship.log.p_action], self.dim)
        else:
            raise BaseException('depositor but ship is on yard pos and didnt switch role?')

    def determine_ship_action(self, ship):
        """Harvest/Deposit cycle"""
        if ship.log.role == 'NEW':
            self.assign_role(ship)
        ship.log.yard = self.get_nearest_shipyard(ship.position)
        # Finished deposit? Return to harvest
        if ship.log.yard is not None:
            if ship.log.role == 'DEP' and ship.position == ship.log.yard.position:
                ship.log.role = 'HVST'
        if ship.log.role == 'HVST':
            if ship.log.spot is None or self.board.cells[ship.log.spot].halite < self.halite_harvest_minimum:
                ship.log.spot = self.determine_best_harvest_spot(ship)
            # Finished harvest? Return to deposit
            if ship.halite > self.ship_carry_maximum:
                ship.log.role = 'DEP'
            else:
                self.determine_best_harvest_action(ship)
        if ship.log.role == 'DEP':
            self.determine_best_deposit_action(ship)

    def get_best_ship_for_yard(self):
        """Return ship with minimum mean distance to others.
        Calculate distance between each point pair.
        Calculate mean of distance for each ships pairings."""
        if len(self.me.ships) == 1:
            return self.me.ships[0]
        else:
            ship_mean_dist = {}
            pair_dists = {}
            for pair in combinations([s.position for s in self.me.ships], 2):
                pair_dists[pair] = self.dist(*pair)
            for ship in self.me.ships:
                ship_mean_dist[ship] = np.mean([dist for pair, dist in pair_dists.items() if ship.position in pair])
        return sorted(ship_mean_dist, key=ship_mean_dist.get)[0]

    def get_actions(self, obs, config):
        """Main loop"""
        self.board = Board(obs, config)
        self.me = self.board.current_player
        me = self.me  # just for shorthand
        spawncount = 0
        self.refresh_ships()
        self.yardcount = len(self.me.shipyards)
        self.mean_halite = int(np.mean([cell.halite for cell in self.board.cells.values() if cell.halite != 0]))
        self.halite_harvest_minimum = self.mean_halite
        self.ship_carry_maximum = self.mean_halite * 4
        self.enemy_ship_points = [ship.position for plr in self.board.players.values()
                                  if plr is not self.me for ship in plr.ships]
        self.generate_harvest_values()
        # Main ship loop - iterate until each ship has an action
        # TODO - order ships by priority. should be done after yard building assigned
        # TODO - ships on SY should go first
        # TODO - if ndocks < 1, prioritize building yard
        self.action_iter = 0
        while len(LOG.set_actions) != len(me.ships):
            self.action_iter += 1
            if self.action_iter > 24:
                raise BaseException(f"action resolution iteration > 24 - probable infinite loop")
            if self.action_iter % 10 == 0:
                print(f"Action Iter:{self.action_iter}")

            # If no yards, create and mark point
            if len(self.me.shipyards) == 0:
                ship = self.get_best_ship_for_yard()
                ship.next_action = ShipAction.CONVERT
                ship.log.set_action = ShipAction.CONVERT
                # conversion is resolved before collision - we don't need to reserve point with log.set_point
                ship.log.p_point = None
                self.prospective_yard = Shipyard('PROSPECTIVE', ship.position, self.me.id, self.board)

            # Calculate best potential actions
            for ship in [s for s in me.ships if s.log.set_action is None]:
                self.determine_ship_action(ship)

            # Confirm non-conflicting actions. Record set actions in ship log to keep track of
            # how many ships actions are finalized.
            p2s = LOG.p_point2ship  # point2ship map - excluding set ships
            for point, ships in p2s.items():
                if len(ships) == 1:  # Only contender - give action
                    ship = ships[0]
                    action, point = ship.log.p_action, ship.log.p_point
                    ship.next_action = action if action != 'WAIT' else None
                    ship.log.set_action, ship.log.set_point = action, point
                    # When ship action is calculated above, any set points should now not be possibilities.
                else:  # Give spot to highest priority ship (currently highest halite)
                    ships_by_halite = sorted([(s, s.halite) for s in ships], key=lambda x: -x[1])
                    priority_ship, halite = ships_by_halite[0]
                    action, point = priority_ship.log.p_action, priority_ship.log.p_point
                    priority_ship.next_action = action if action != 'WAIT' else None
                    priority_ship.log.set_action, priority_ship.log.set_point = action, point

        # Ship building
        for shipyard in me.shipyards:
            # TODO - if enemy ship is adj yard, forgo yard reserve to spawn a ship
            reserve = config.convertCost
            # If we can afford spawn, considering cumulation of other SY spawns and keeping a reserve for one yard.
            have_enough_halite = (me.halite - spawncount * config.spawnCost - reserve) >= config.spawnCost
            should_still_spawn = True  # TODO enable after fixing issues - sum([c.halite for c in self.board.cells.values()])/4 > 2*self.board.configuration.spawn_cost
            no_ship_reserved_point = shipyard.position not in LOG.set_points
            if have_enough_halite and should_still_spawn and no_ship_reserved_point and self.keep_spawning_tripswitch:
                shipyard.next_action = ShipyardAction.SPAWN
                spawncount += 1
        self.board_prev = self.board
        return me.next_actions


def agent(obs, config):
    global myBot
    if 'myBot' not in globals():
        myBot = {obs.player: MyAgent(obs, config)}
    # myBot[obs.player].update(obs, config)
    actions = myBot[obs.player].get_actions(obs, config)
    return actions
