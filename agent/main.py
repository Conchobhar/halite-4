from collections import OrderedDict
from functools import lru_cache
from itertools import combinations, product, count, chain
import numpy as np
import random
import time
from kaggle_environments.envs.halite.helpers import Board, ShipAction, ShipyardAction, Ship, Shipyard, Point

np.seterr(all='raise')

"""Items for ship log
    'role'       What is ships purpose?
    'spot'       Where is ship focused on?
    'p_action'   potential action
    'p_point'    potential resultant point (from action)
    'set_action' Confirmed move action
    'set_point'  Confirmed resultant point (from action)
"""

# Magic numbers - moved most other constants to attributes on agent
magic = {
    'early_game_step': 90,
    'late_game_step': 250,  # Change ship behaviour for end game
    'evade_ship_count': 50,  # How many ships any opponent needs to engage evasion
}

# Action delta - Map move action to positional change
adelta = {
    ShipAction.NORTH: (0, 1),
    ShipAction.EAST: (1, 0),
    ShipAction.SOUTH: (0, -1),
    ShipAction.WEST: (-1, 0),
    'WAIT': (0, 0),
}

# Action inverse - Map move action to its opposite
ainverse = {
    ShipAction.NORTH: ShipAction.SOUTH,
    ShipAction.EAST: ShipAction.WEST,
    ShipAction.SOUTH: ShipAction.NORTH,
    ShipAction.WEST: ShipAction.EAST,
    'WAIT': 'WAIT',
}

# PlayerID to starting position
pid2quadrant = {
    0: Point(5, 15),
    1: Point(15, 15),
    2: Point(5, 5),
    3: Point(15, 5)
}
# PlayerID to Nonadjacent player id
pid2nonadj = {
    0: 3,
    1: 2,
    2: 1,
    3: 0,
}
# PlayerID to adjacent player ids
pid2adj = {
    0: (1, 3),
    1: (0, 3),
    2: (0, 3),
    3: (2, 1),
}


"""ROLES 
Attributes:
    is_tracking - Does this role track an id that might not persist across turns? Excluding own shipyards.
    is_expendable - Is this role expendable for certain tasks? Only used for supressors.
    is_yardable - Is this role suitable for yard conversion.

Descriptions:
    NEW
        Newly spawned
    HVST
        Harvest halite
    DEP
        Deposit halite
    DFND
        Defend a specific shipyard
    assualt
        Attack a specific enemy shipyard
    call_home
        end game deposit - not used
    strike_yards
        end game assualt
    evade
        Only try to survive
    suppressor
        Chase enemy ships and try to win collisions
    emergancy
        role for situation of zero shipyards and not enough halite to spawn a yard
"""
ROLES = {
    'NEW': {'is_tracking': False, 'is_expendable':True, 'is_yardable': True},
    'HVST': {'is_tracking': False, 'is_expendable':True, 'is_yardable': True},
    'DEP': {'is_tracking': False, 'is_expendable':True, 'is_yardable': True},
    'DFND': {'is_tracking': True, 'is_expendable':False, 'is_yardable': False},
    'assault': {'is_tracking': True, 'is_expendable':False, 'is_yardable': False},
    'call_home': {'is_tracking': False, 'is_expendable':True, 'is_yardable': True},
    'strike_yards': {'is_tracking': True, 'is_expendable':False, 'is_yardable': True},
    'evade': {'is_tracking': False, 'is_expendable':True, 'is_yardable': True},
    'suppressor': {'is_tracking': True, 'is_expendable':True, 'is_yardable': False},
    'emergancy': {'is_tracking': True, 'is_expendable':True, 'is_yardable': True},
}

# Define required conditions for a role to move into a specific spot, and the priority order
# Priority defined in order, actual key value not used
role2conditions = {
    'HVST': OrderedDict({
        'is_not_occupied_by_threat_excluding_eq': -2,
        'is_not_occupied_by_potential_threats_excluding_eq': -2,
        'is_not_occupied_by_self': 1,
        'is_not_occupied_by_threat': 0,
        'is_not_occupied_by_enemy_yard': 2,
        'is_not_occupied_by_potential_threats': 3,
    }),
    'DEP': OrderedDict({
        'is_not_occupied_by_threat_excluding_eq': -2,
        'is_not_occupied_by_potential_threats_excluding_eq': -2,
        'is_not_occupied_by_self': 1,
        'is_not_occupied_by_threat': 0,
        'is_not_occupied_by_enemy_yard': 2,
        'is_not_occupied_by_potential_threats': 3,
    }),
    'DFND': OrderedDict({
        'is_not_occupied_by_threat_excluding_eq': -2,
        'is_not_occupied_by_potential_threats_excluding_eq': -2,
        'is_not_occupied_by_self': 1,
        'is_not_occupied_by_threat': 0,
        'is_not_occupied_by_enemy_yard': 2,
        'is_not_waiting_on_halite': 3,
    }),
    'assault': OrderedDict({
        'is_not_occupied_by_threat_excluding_eq': -2,
        'is_not_occupied_by_potential_threats_excluding_eq': -2,
        'is_not_occupied_by_self': 1,
        'is_not_occupied_by_threat': 0,
        'is_not_waiting_on_halite': 2,
        'is_not_occupied_by_potential_threats': 3,
    }),
    'call_home': OrderedDict({
        'is_not_occupied_by_threat_excluding_eq': -2,
        'is_not_occupied_by_potential_threats_excluding_eq': -2,
        'is_not_occupied_by_self': 1,
        'is_not_occupied_by_threat': 0,
        'is_not_waiting_on_yard': 2,  # Don't block yard at call home!
        'is_not_occupied_by_enemy_yard': 3,
        'is_not_occupied_by_potential_threats': 4,
    }),
    'evade': OrderedDict({
        'is_not_occupied_by_threat_excluding_eq': -2,
        'is_not_occupied_by_potential_threats_excluding_eq': -2,
        'is_not_occupied_by_self': 1,
        'is_not_occupied_by_threat': 0,
        'is_not_occupied_by_enemy_yard': 2,
        'is_not_occupied_by_potential_threats': 3,
    }),
    'strike_yards': OrderedDict({
        'is_not_occupied_by_self': 1,
        'is_not_occupied_by_threat_excluding_eq': -2,
        'is_not_occupied_by_potential_threats_excluding_eq': -2,
    }),
    'suppressor': OrderedDict({
        'is_not_occupied_by_self': -1,
        'is_not_occupied_by_threat_excluding_eq': -2,
        'is_not_occupied_by_potential_threats_excluding_eq': -2,
        'is_not_waiting_on_halite': 0,
        'is_not_occupied_by_enemy_yard': 1,
        'is_not_occupied_by_threat': 2,
        'is_not_occupied_by_potential_threats': 3,
    }),
    'emergancy': OrderedDict({
        'is_not_occupied_by_threat_excluding_eq': -2,
        'is_not_occupied_by_potential_threats_excluding_eq': -2,
        'is_not_occupied_by_self': 1,
        'is_not_occupied_by_threat': 0,
        'is_not_occupied_by_enemy_yard': 2,
        'is_not_occupied_by_potential_threats': 3,
    }),
}


def dist(p1, p2, dim=21):
    """Calculate the shortest L1 norm/ manhatten distance between two points.
    By taking the min of (coord, dim - coord) we account for the cyclic nature of the map."""
    p = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
    x, y = min(p[0], dim - p[0]), min(p[1], dim - p[1])
    return abs(sum([x, y]))


def midpoint(p1, p2, dim=21):
    """Calculate the closest point equidistant from two other points.
    If the absolute distance between two coords is less than half the board dimension, then
    this is just the mean of two points, else we need to consider looping."""
    x = (p1[0] + p2[0])//2 if abs(p1[0] - p2[0]) < dim//2 else ((p1[0] + p2[0]) + dim)//2 % dim
    y = (p1[1] + p2[1]) // 2 if abs(p1[1] - p2[1]) < dim // 2 else ((p1[1] + p2[1]) - dim) // 2 % dim
    return (x, y)


# Init lookup dicts - thought this would solve timeout issues but ended up not being the problem
lookup = {
    'dist': {},
    'midpoint': {},
}


def get_dist(p1, p2):
    """Lookup value for distance"""
    return lookup['dist'][tuple(sorted((p1, p2)))]


def get_midpoint(p1, p2):
    """Lookup value for midpoint"""
    return lookup['midpoint'][tuple(sorted((p1, p2)))]


def get_radial_grid(r):
    """Get coordinate grid for all points within a radius r centered at (0,0)
    Can get this by taking product of coords where
        sum of abs values is <= radius"""
    coords = [x for x in range(-r, r + 1)]
    adjs = [x for x in product(coords, coords) if sum([abs(c) for c in x]) <= r]
    return adjs


@lru_cache(maxsize=None)
def get_adjacent_positions(p, r, dim=21):
    adjs = get_radial_grid(r)
    adjs.remove((0, 0))
    return [((p + x) % dim) for x in adjs]


def precompute():
    """Precompute lookup dicts"""
    coords = product(range(0, 21), repeat=2)
    coord_pairs = [tuple(sorted(pair)) for pair in product(coords, repeat=2)]
    coord_pairs_set = set(coord_pairs)
    for pair in coord_pairs_set:
        lookup['dist'][pair] = dist(*pair)
        lookup['midpoint'][pair] = midpoint(*pair)
    coords = product(range(0, 21), repeat=2)
    for coord in coords:
        for r in range(1,8):
            _ = get_adjacent_positions(Point(*coord), r)


class Squadron:
    """Collection of ships with coordinated objective."""

    global LOG, myBot, myId

    def __init__(self, squad_id):
        self.squad_id = squad_id
        self.ship_ids = set()
        self.yard = None
        self.swarm_centroid = None
        self.swarm_radius = None
        self.target_id = None  # Objective entity
        self.previous_pair = None
        self.midpoint = None  # centre of mass
        self.agent = myBot[myId]

    def __repr__(self):
        targpos = self.target_entity.position if self.target_entity is not None else 'NONE'
        return f"{'{'}ID:{self.squad_id}, ns:{len(self.ship_ids)}, mid:{self.midpoint}, targ:{targpos}{'}'}"

    @property
    def nships(self):
        return len(self.ship_ids)

    @property
    def target_entity(self):
        if self.target_id in LOG.id2obj:
            return LOG.id2obj[self.target_id]
        else:
            return None

    def assign(self, ship):
        """Keep track of ids assigned to squadron"""
        self.ship_ids.add(ship.id)

    def update_midpoint(self):
        """Calculate centroid of squadron. For more than 2 ships, we get a cheap approximation."""
        if len(self.ship_ids) == 0:
            self.midpoint = 0
        elif len(self.ship_ids) == 1:
            self.midpoint = LOG.id2obj[list(self.ship_ids)[0]].position
        else:  # Running average over sequential ship pairs
            ships = [LOG.id2obj[sid] for sid in self.ship_ids]
            self.midpoint = ships[0].position
            for ship in ships[1::]:
                self.midpoint = get_midpoint(ship.position, self.midpoint)
        self.yard = self.agent.get_nearest_shipyard(self.midpoint)

    def refresh_entities(self):
        """Remove ids no longer in game or if assigned a different role."""
        for sid in self.ship_ids.copy():
            if sid not in LOG.id2obj:
                self.ship_ids.remove(sid)
            elif LOG.id2obj[sid].log.role != 'suppressor':
                self.ship_ids.remove(sid)
        if self.target_id not in LOG.id2obj:
            self.target_id = None

    def attach_squad_to_ships(self):
        """Update each ship log with reference to squadron"""
        for sid in self.ship_ids:
            LOG.id2obj[sid].log.squad = self

    def get_best_target_id_by_dist(self):
        """Prioritize targets by:
            opportunistically between two squads
            ship that is min of mean of distance to my ship yard and distance to squad
            Take non-zero halite ships first, then zero, then go for yards"""
        active_squads = [sq for sq in LOG.squadrons if sq.target_id is not None]
        n_active = len(active_squads)
        if n_active > 0:  # Opportunistically take flanked targts
            for sq in active_squads:
                mid_point = get_midpoint(self.midpoint, sq.midpoint)
                sq2targ_dist = get_dist(self.midpoint, sq.target_entity.position)
                targ2midpoint_dist = get_dist(mid_point, sq.target_entity.position)
                if targ2midpoint_dist < 3 and sq2targ_dist < 3:
                    return sq.target_id
        potential_targs = {}
        zero_targs = {}
        for es in self.agent.enemy_ships:
            mdist = (get_dist(es.position, self.yard.position) + get_dist(es.position, self.midpoint)**0.5)/2
            if es.halite > 0:
                potential_targs[es] = mdist
            else:
                zero_targs[es] = mdist
        if len(potential_targs) > 0:
            targ = min(potential_targs, key=potential_targs.get)
        elif len(zero_targs) > 0:
            targ = min(zero_targs, key=zero_targs.get)
        else:  # No ships left? Go for yards
            targ = self.agent.enemy_yards[0]
        return targ.id

    def reevaluate_target_id(self):
        """Determine if we should go after a better target i.e. if
            target is too close to its own yard
            target is too far from my own yards"""
        targ = LOG.id2obj[self.target_id]
        esy = self.agent.get_nearest_enemy_shipyard(targ)
        asy = self.agent.get_nearest_shipyard(targ.position)
        cond_reeval = (targ.__class__ is Ship
                       and (
                               targ.halite == 0
                               or False if esy is None else get_dist(targ.position, esy.position) < 2
                               or False if asy is None else get_dist(targ.position, asy.position) > 8)
                       )
        if cond_reeval:
            return self.get_best_target_id_by_dist()
        else:
            return self.target_id

    def refresh_targets(self):
        """Get or reevaluate target id.
        Swarm centroid and radius are no longer used."""
        if len(self.agent.enemy_ships) > 0:
            self.swarm_centroid = self.agent.max_cargo_density_pos
            self.swarm_radius = 4
        elif len(self.agent.best_suppress_yards) > 0:  # Get a random yard target unless already assigned one.
            syposs = [sy.position for sy in self.agent.best_suppress_yards]
            if self.swarm_centroid not in syposs:
                self.swarm_centroid = random.choice(syposs)
            self.swarm_radius = 4
        else:  # Unlikely case
            self.swarm_centroid = Point(10,10)
            self.swarm_radius = 10
        # Has target already and it still exists?
        if self.target_id is not None and self.target_id in LOG.id2obj:
            self.target_id = self.reevaluate_target_id()
        # Needs target?
        if self.target_id is None:
            self.target_id = self.get_best_target_id_by_dist()
        else:
            # Had target but it was destroyed?
            self.target_id = self.get_best_target_id_by_dist()


class LogEntry:
    """Log values for each ship. Like a dictionary, without the ['fluff']"""

    def __init__(self):
        self.role = 'NEW'
        self.role_suspended = None
        self.target_cell = None   # Where ship wants to move based on role
        self.spot = None            # assigned harvest spot
        self.yard = None            # nearerst yard
        self.squad = None
        self.p_action = None
        self.p_point = None
        self.set_action = None
        self.set_point = None
        self.last_action = None
        self.frustration = 0
        self.is_frustrated = False
        self.adj_allies = 0
        self.adj_threats = 0
        self.track_id = None
        self.resetable_names = ['p_action', 'p_point', 'set_action', 'set_point', ]

    def reset_turn_values(self):
        """Reset values that don't carry across turns."""
        for name in self.resetable_names:
            setattr(self, name, None)

    def __str__(self):
        return f"R:{self.role} S:{self.spot} p_a:{self.p_action} p_p:{self.p_point}"


class Log(dict):
    """Super log to keep track of information across all ships.
    Keys are ships. Values are a LogEntry()"""

    def __init__(self):
        super().__init__()
        # Keep a reference to me - necessary to extract `next_actions`
        self.board = None
        self.me = None
        self.harvest_spot_values = None
        self.enemy_targ_ids = None
        self.id2obj = None  # Map game id's to the objects regenerated each turn
        self.tracking_roles = [role for role, attr in ROLES.items() if attr['is_tracking']]
        self.expendable_roles = [role for role, attr in ROLES.items() if attr['is_expendable']]
        self.yardable_roles = [role for role, attr in ROLES.items() if attr['is_yardable']]
        self.squadrons = []

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
    def set_actions(self):  # Assigned actions
        return [x.set_action for x in self.values() if x.set_action is not None]

    @property
    def free_spots_by_value(self):  # Harvestable spots not yet assigned
        return [(spot, value) for spot, value in self.harvest_spot_values if spot not in self.spots]

    def point2potential_ships(self, shipset):
        """Mapping of points to the ships which want to move there next turn.
        Excluding ships already set"""
        p2s = {}
        for ship in [s for s in shipset if s.log.set_action is None]:
            if ship.log.p_point not in p2s:
                p2s[ship.log.p_point] = [ship]
            else:
                p2s[ship.log.p_point].append(ship)
        return p2s

    @property
    def unset_ships(self):  # Assigned next turn positions
        return [s for s, log in self.items() if log.set_point is None]

    @property
    def yard2defenders(self):
        """Map of shipyards to the defender assigned to it"""
        return {self.id2obj[le.track_id]: s for s, le in self.items() if s.log.role == 'DFND'}

    @property
    def defenders2yard(self):
        """Inverse of yard2defenders"""
        return {v: k for k, v in self.yard2defenders.items()}

    @property
    def yard2assaulters(self):
        """Map of shipyards to the assualters assigned to it"""
        return {self.id2obj[le.track_id]: s for s, le in self.items() if s.log.role == 'assault'}

    def get_assaulters(self):
        """Inverse of yard2assualters"""
        return [s for s, log in self.items() if log.role == 'assault']

    def get_suppressors(self):
        return [s for s, log in self.items() if log.role == 'suppressor']

    def get_id2suppressors(self): # not used
        id2s = {}
        for ship in [s for s in self if s.log.role == 'suppressor']:
            if ship.log.track_id not in id2s:
                id2s[ship.log.track_id] = [ship]
            else:
                id2s[ship.log.track_id].append(ship)
        return id2s

    @property
    def get_squaddie_ids(self):
        """Get set of all assigned ship ids"""
        return set(chain(*[sq.ship_ids for sq in self.squadrons]))

    def squadrons_centroid(self):
        pass
        # return [sq for sq in self.squadrons if sq.target_id is not None and ]


global LOG
LOG = Log()


class MyAgent:

    def __init__(self, obs, config):
        self.board = Board(obs, config)
        self.board_prev = None
        self.config = self.board.configuration
        self.me = self.board.current_player
        self.pids_adj = pid2adj[self.me.id]
        self.pid_nonadj = pid2nonadj[self.me.id]
        self.enemy_pids = self.pids_adj + (self.pid_nonadj,)
        self.dim = config.size
        self.mid = config.size // 2
        self.quadrant_position = pid2quadrant[self.me.id]
        self.quadrant_points = self.get_quadrant_points()
        self.quadrant_adjs = set(self.get_adjs(self.quadrant_position, r=1))
        self.should_still_spawn = None
        self.harvest_spot_values = None
        self.enemy_ship_points = None
        self.p2net_halite = {}
        self.halite_global_mean = None
        self.halite_global_median = None
        self.halite_global_std = None
        self.halite_density = {}
        self.cargo_density = {}
        self.cargo_density_closest = {}
        self.suppress_value = {}
        self.suppress_value_density = {}
        self.frustration_max = 3
        self.global_min_requirement_to_harvest = None
        self.global_min_requirement_to_harvest_locally = None
        self.spawncount = None
        self.convertcount = None
        self.yardcount = None
        self.prospective_yard = None
        self.is_yard_emergancy = False
        self.action_iter = None
        self.ship_carry_maximum = None  # calculated at start of each loop
        self.halite_harvest_minimum = None
        self.role2conditions = role2conditions
        self.squadrons = []

    @property
    def halite_balance(self):
        """Halite left after considering set expenditures"""
        return (self.me.halite
                - self.spawncount * self.board.configuration.spawn_cost
                - self.convertcount * self.board.configuration.convert_cost)

    def can_afford_to_spawn(self):
        return self.halite_balance >= self.board.configuration.spawn_cost

    @staticmethod
    def maximum_squadrons():
        """Can have a squadron for every 3 suppressors"""
        lsup = len(LOG.get_suppressors())
        return 1 + (lsup - 1)//3

    def get_quadrant_points(self):
        """Define player quadrant incl. shared points."""
        points = []
        qp = (self.mid // 2) + 1
        for x in range(-qp, qp + 1):
            for y in range(-qp, qp + 1):
                points.append(self.quadrant_position.translate(Point(x, y), self.dim))
        return points

    def count_adjacent_allies(self, ship, r=1):
        """Adjacent ally ships within r"""
        c = 0
        for adjc in self.get_adjs(ship.position,r=r, return_cells=True):
            if adjc.ship is not None and adjc.ship.player_id == self.me.id:
                c += 1
        return c

    def count_adjacent_threats(self, ship, r=1, is_equal_threat=True):
        """Adjacent threats within r"""
        c = 0
        if is_equal_threat:
            for adjc in self.get_adjs(ship.position, r=r, return_cells=True):
                if adjc.ship is not None and adjc.ship.player_id != self.me.id and adjc.ship.halite < ship.halite:
                    c += 1
        else:
            for adjc in self.get_adjs(ship.position,r=r, return_cells=True):
                if adjc.ship is not None and adjc.ship.player_id != self.me.id and adjc.ship.halite <= ship.halite:
                    c += 1
        return c

    def refresh_log(self):
        """Attach `me` to LOG
        Refresh log keys (i.e. ship obj) connecting to new state through ship.id
        Attach global LOG as attribute for conveniance."""
        global LOG
        old_log = LOG
        ids = {s.id: le for s, le in old_log.items()}
        LOG = Log()  # Clears any crashed ships - LOG is rebuilt below
        LOG.id2obj = {id:obj for id,obj in {*self.board.ships.items(), *self.board.shipyards.items()}}
        LOG.me = self.me
        LOG.board = self.board
        LOG.harvest_spot_values = self.harvest_spot_values

        for s in self.me.ships:
            # Default .next_action is None, which implies waiting.
            # Use this default to indicate that we have not yet decided on an action.
            s.next_action = 'NOTSET'
            if s.id in ids:  # reconnect log
                LOG[s] = ids[s.id]
                s.log = LOG[s]
                s.log.reset_turn_values()
                if s.log.track_id is not None and s.log.track_id not in LOG.id2obj:  # If trackable is gone reset role and track_id
                    s.log.role = 'HVST'   # Should keep role assignment in one place...
                    s.log.track_id = None

            else:  # log new ships
                LOG[s] = LogEntry()
                s.log = LOG[s]
            s.log.adj_allies = self.count_adjacent_allies(s, r=1)
            s.log.adj_threats = self.count_adjacent_threats(s, r=2, is_equal_threat=False)
            s.log.adj_threats_immediate = self.count_adjacent_threats(s, r=1, is_equal_threat=False)

        # Squadron carry over handled in assemble_squadrons()
        LOG.squadrons = old_log.squadrons

    def dist(self, p1, p2):
        """Alias to avoid refrac"""
        return get_dist(p1, p2)

    def get_nearest_shipyard(self, pos):
        """Get nearest allied shipyard to position."""
        if self.yardcount == 0:
            if self.prospective_yard is not None:
                return self.prospective_yard
            else:
                return Shipyard('EMERGANCY', (15, 15), self.me.id, self.board)
        if self.yardcount == 1:
            return self.me.shipyards[0]
        else:
            dist2yards = [(self.dist(pos, sy.position), sy) for sy in self.me.shipyards]
            return min(dist2yards, key=lambda x: x[0])[1]

    def determine_best_harvest_spot_locally(self, ship):
        """For all adjacent cells within r=2, determine best cell by halite and considering nearby enemies."""
        d = {}
        ship_yard_dist = self.dist(ship.position, ship.log.yard.position)
        for pos in self.get_adjs(ship.position, r=2):
            cell = self.board.cells[pos]
            is_not_on_prospecive_yard = not (self.prospective_yard is not None and cell.position == self.prospective_yard.position)
            if cell.shipyard is None and is_not_on_prospecive_yard and cell.halite > self.global_min_requirement_to_harvest_locally:
                threat_array = [(c.ship is not None and c.ship.player_id != self.me.id and c.ship.halite <= ship.halite)
                                for c in self.get_adjs(ship.position, r=1, return_cells=True)]
                threat_factor = 1 - sum(threat_array)/len(threat_array)
                # Negatively weight moving away from yard, but no benefit for moving towards it
                dist = max(self.dist(ship.log.yard.position, pos), ship_yard_dist)
                d[pos] = (self.board.cells[pos].halite / dist) * threat_factor
        d = sorted(d, key=d.get)[::-1]
        for pos in d:
            if pos not in LOG.spots and not self.is_pos_occupied_by_threat(ship, pos):
                return pos
        return None

    def determine_best_harvest_spot_from_yard(self, ship):
        """Choose a spot to harvest - values already sorted desceding."""
        if ship.log.spot is not None:
            ship_spot2w = [(p,w) for p,w in self.harvest_spot_values if p == ship.position]
            ship_spot, ship_spotweight = ship_spot2w[0]
            spots2weight = sorted(LOG.free_spots_by_value + ship_spot2w, key=lambda x: -x[1])
            for spot, weight in spots2weight:
                if (weight > ship_spotweight and self.dist(ship.position, spot) <= self.dist(ship.position, ship_spot)):
                    return spot
            return ship.log.spot

        spots2weight = sorted(LOG.free_spots_by_value, key=lambda x: -x[1])
        for spot, weight in spots2weight:
            cell = self.board.cells[spot]
            if cell.halite > self.global_min_requirement_to_harvest:
                return spot
        # Share spots in this case
        spots_with_min_halite = [spot for spot, value in self.harvest_spot_values]
        for spot in spots_with_min_halite:
            return spot

    def map_cyclic_coords(self, x):
        """Map higher half of coordinate space to its -ve equivalent
        e.g. for board dimension of length 5:
            (0,1,2,3,4,5) --> (0,1,2,-2,-1,0)"""
        return (x + self.mid) % self.dim - self.mid

    def is_pos_occupied_by_threat(self, ship, ppos, is_equal_threat=True):
        """Determine if position is occupied by a threat.
        Don't consider a ship with equal halite a threat unless depositing."""
        cell = self.board.cells[ppos]
        ppos_ship = cell.ship
        if ppos_ship is not None:
            if ship.log.role == 'DEP' or is_equal_threat:
                is_occupied_by_threat = (
                        ppos_ship.player_id != ship.player_id and ppos_ship.halite <= ship.halite)
            else:
                is_occupied_by_threat = (
                        ppos_ship.player_id != ship.player_id and ppos_ship.halite < ship.halite)
        else:
            is_occupied_by_threat = False
        return is_occupied_by_threat

    def get_nearest_enemy_shipyard(self, ship):
        p = self.board.players[ship.player_id]
        if len(p.shipyards) > 0:
            return min(p.shipyards, key=lambda sy: self.dist(ship.position, sy.position))
        else:
            return None

    def predict_enemy_move_to_target(self, ship):
        """Basic enemy model"""
        if ship.halite > 0:
            if ship.halite < 200:  # Harvesting
                cells = [adjc for adjc in self.get_adjs(ship.position, r=1, return_cells=True)]
                ship_targ = max(cells, key=lambda c:c.halite).position
            else:  # Depositing
                esy = self.get_nearest_enemy_shipyard(ship)
                ship_targ = esy.position if esy is not None else ship.position
        else:
            ship_targ = ship.position
        best_to_worst_actions = self.get_best2worst_actions_enemy(ship, ship_targ)
        baction = best_to_worst_actions[0]
        return ship.position.translate(adelta[baction], self.dim)

    def get_best2worst_actions_enemy(self, ship, pt):
        """Basic enemy model actions"""
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
        pos_adjs = [ship.position.translate(adelta[a], self.dim)
                     for a in actions if a not in (None,)]
        if any([self.is_pos_occupied_by_threat(ship, pos_adj) for pos_adj in pos_adjs]):
            actions['WAIT'] += -2.1
        for action in actions:
            ppos = ps.translate(adelta[action], self.dim)
            cell = self.board.cells[ppos]
            if cell.ship is not None and cell.ship.player_id != ship.player_id:
                actions[action] +=  -2.1
        best_to_worst_actions = sorted(actions, key=actions.get)[::-1]
        return best_to_worst_actions

    def get_best2worst_actions(self, ship, pt, weighted=True, no_safe_moves=False):
        """Get the set of possible actions, ordered best to worst based on decreasing distance to target.
        Weighted by adj negative and positive collisions. This got messy"""
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
        if not weighted:
            return sorted(actions, key=actions.get)[::-1]
        pos_adjs = [ship.position.translate(adelta[a], self.dim)
                     for a in actions if a not in (None,)]
        if no_safe_moves:  # negatively weight collisions
            for action in actions:
                ppos = ps.translate(adelta[action], self.dim)
                action_inverse = ainverse[action]
                pcell_adjs = [self.board.cells[ppos.translate(adelta[a], self.dim)] for a in actions if
                              a not in (action_inverse,)]
                n_threats = len([c for c in pcell_adjs if
                              c.ship is not None and c.ship.player_id != self.me.id and c.ship.halite <= ship.halite])
                actions[action] = actions[action] - 2*n_threats
            return sorted(actions, key=actions.get)[::-1]
        if ship.log.role != 'DFND' and any([self.is_pos_occupied_by_threat(ship, pos_adj) for pos_adj in pos_adjs]):
            # Waiting is worst choice if any adjacent threats. If also frustrated, WAIT will
            # still rank worst choice.
            actions['WAIT'] -= 2
        if ship.log.last_action is not None and ship.log.is_frustrated:
            # Don't wait about, and don't undo the last move. Arbitrary large decrements
            actions['WAIT'] -= 2
            if ship.log.last_action is not ShipAction.CONVERT:
                actions[ainverse[ship.log.last_action]] -= 3
        # Adjust action weight by:
        #   number of potential positive collisions - modifier range 0, +1
        #   general entity density in area - modifier between -.5, 0
        # +ve collisions more enticing that the density they add
        for action in actions:
            ppos = ps.translate(adelta[action], self.dim)
            action_inverse = ainverse[action]
            pcell_adjs = [self.board.cells[ppos.translate(adelta[a], self.dim)] for a in actions if a not in (action_inverse, )]
            n_pcol = len([c for c in pcell_adjs if c.ship is not None and c.ship.player_id != self.me.id and c.ship.halite > ship.halite])
            actions[action] = actions[action] + n_pcol/4 - self.ship_and_yard_density[ppos]/2
        best_to_worst_actions = sorted(actions, key=actions.get)[::-1]
        maybe_yard = self.board.cells[ps].shipyard
        actions['WAIT'] = actions['WAIT'] - 1 if maybe_yard is not None else actions['WAIT']  # Penalty for waiting about on yards
        return best_to_worst_actions

    def move_to_target(self, ship, pt):
        """Normalize coordinates and determine best action for approaching target.
        ship - ship moving
        pt - pos of target
        ps - pos of ship
        Normalize:  translate origin to ps (i.e. subtract ps from pt)
                    map higher half coords to -ve values
        Avoid actions that are potentially unsafe, but iteratively
        lessen restrictions until first 'safe' move is found.

        Ideally would rate every option based on threats vs potential to win encounter.
        This is a vestigial mess that should have been changed after the first few successful bots.
        """
        ps = ship.position
        best_to_worst_actions = self.get_best2worst_actions(ship, pt)
        best_action = best_to_worst_actions[0]
        chosen_action = 'UNDECIDED'
        cond_iter, conditions = 0, {}
        role_conditions = self.role2conditions[ship.log.role]
        while chosen_action == 'UNDECIDED':
            # For each possible action, in order of preference, determine if safe
            # If no action is safe, reduce the amount safety conditions until no options are left
            for action in best_to_worst_actions:
                ppos = ps.translate(adelta[action], self.dim)
                action_inverse = ainverse[action]
                ppos_adjs = [ppos.translate(adelta[a], self.dim) for a in best_to_worst_actions if a not in (None, action_inverse)]
                cell = self.board.cells[ppos]
                is_my_yard_and_spawning = cell.shipyard is not None and cell.shipyard.player_id == self.me.id and cell.shipyard.next_action is not None
                is_target_and_esy = ppos == pt and cell.shipyard is not None and cell.shipyard.player_id != self.me.id
                # Take riskier moves near own yard
                is_near_own_yard = self.dist(ship.position, ship.log.yard.position) <= 3
                is_in_wake_of_my_threatening_ship = False
                if cell.ship is not None and cell.ship.player_id == self.me.id:
                    cship = cell.ship
                    is_in_wake_of_my_threatening_ship = all([not self.is_pos_occupied_by_threat(cship, pos, is_equal_threat=False) for pos in self.get_adjs(ppos,1)])
                cell_behind = self.board.cells[ppos.translate(adelta[action], self.dim)]
                behind_threat = False
                if cell_behind.ship is not None and cell_behind.ship.player_id != self.me.id and cell_behind.ship.halite <= ship.halite:
                    behind_threat = True
                # not occupied by enemy ship with less halite
                ship_eq_0 = False #ship.halite == 0
                conditions['is_not_occupied_by_threat_excluding_eq'] = not self.is_pos_occupied_by_threat(ship, ppos, is_equal_threat=False) or not behind_threat
                conditions['is_not_occupied_by_threat'] = not self.is_pos_occupied_by_threat(ship, ppos) or not behind_threat
                conditions['is_not_occupied_by_self'] = (ppos not in LOG.set_points and not is_my_yard_and_spawning)
                conditions['is_not_waiting_on_yard'] = (True if action != 'WAIT' else
                                                        ship.log.role != 'DEP' and ppos not in [sy.position for sy in self.me.shipyards])
                conditions['is_not_waiting_on_halite'] = action != 'WAIT' or cell.halite == 0
                conditions['is_not_occupied_by_enemy_yard'] = True if is_target_and_esy else not (cell.shipyard is not None and cell.shipyard.player_id != self.me.id)
                conditions['is_not_occupied_by_potential_threats_excluding_eq'] = ship_eq_0 or is_in_wake_of_my_threatening_ship or all(
                    [not self.is_pos_occupied_by_threat(ship, ppos_adj, is_equal_threat=False) for ppos_adj in ppos_adjs])
                conditions['is_not_occupied_by_potential_threats'] = ship_eq_0 or is_in_wake_of_my_threatening_ship or all(
                    [not self.is_pos_occupied_by_threat(ship, ppos_adj) for ppos_adj in ppos_adjs])
                # Conditions are ordered by priority
                is_met = [conditions[cond] for cond in role_conditions]
                cond_lim = len(is_met) - cond_iter
                if all(is_met[0:cond_lim]):
                    chosen_action = action
                    break
            cond_iter += 1
            if cond_lim == 0:
                best_to_worst_actions = self.get_best2worst_actions(ship, pt, no_safe_moves=True)
                best_action = best_to_worst_actions[0]
                chosen_action = best_action  # No safe moves found
                break
        return chosen_action

    def get_spot_with_least_threat(self, ship):
        """Threat is count of threatening ships within some large radius.
        Don't consider waiting on current pos - want to maintain minimum halite"""
        pos2threat = {}
        for pos in self.get_adjs(ship.position, r=1):
            pos2threat[pos] = sum([(c.ship is not None and c.ship.player_id != self.me.id and c.ship.halite <= ship.halite)
                                   for c in self.get_adjs(pos, r=5, return_cells=True)])
        return min(pos2threat, key=pos2threat.get)

    def suppress_entity(self, ship):
        """Anticipate targets next position."""
        entity = ship.log.squad.target_entity
        if entity is None:
            print(f"WARNING: Suppress entity for {ship.id}@{ship.position} is None.")
            return Point(10, 10)
        return entity.position  # TODO - needs improved enemy model.
        if entity.__class__ is Ship:
            predicted_target_position = self.predict_enemy_move_to_target(entity)
        else:  # Must be shipyard
            predicted_target_position = entity.position
        return predicted_target_position

    def determine_target_cell(self, ship):
        """Get target cell given ship role."""
        if ship.log.role in ('HVST', 'emergancy'):
            if ship.log.adj_threats_immediate > 0:
                return ship.log.yard.position
            else:
                return ship.log.spot
        elif ship.log.role == 'DEP':
            if ship.log.spot is None:
                return ship.log.yard.position
            else:
                return ship.log.spot
        elif ship.log.role == 'DFND':  # Get to zero halite, move to yard if enemy adjacent
            sy = LOG.defenders2yard[ship]
            return sy.position
        elif ship.log.role == 'assault':
            if ship.halite > 0:
                return ship.log.yard.position
            else:
                return LOG.id2obj[ship.log.track_id].position
        elif ship.log.role == 'call_home':
            if len(self.me.ships) < 5:
                return self.get_spot_with_least_threat(ship)
            return ship.log.yard.position

        elif ship.log.role == 'strike_yards':
            target_id = self.get_attack_target(ship)
            if target_id is None:
                return self.get_spot_with_least_threat(ship)
            else:
                ship.log.track_id = target_id
                return LOG.id2obj[ship.log.track_id].position

        elif ship.log.role == 'evade':  # Deposit to maximize own threat, else move to spot with minimal threat
            if ship.halite > 0:
                return ship.log.yard.position
            else:
                return self.get_spot_with_least_threat(ship)
        elif ship.log.role == 'suppressor':
            if ship.halite > 0:
                return ship.log.yard.position
            else:
                return self.suppress_entity(ship)
        else:
            raise BaseException(f'Need to define logic for new role: {ship.log.role}')

    def is_overharvested_locally(self, ship, spot):
        """Is
            spot **n harvests below local mean?
            ship adj and cell has adj threats?
            """
        # **n harvests under local mean? OR if nearby threat on spot? try somewhere else
        target_cell = self.board.cells[spot]
        ppos_adjs = [ppos for ppos in self.get_adjs(target_cell.position, r=1)]
        is_near_own_yard = self.dist(ship.position, ship.log.yard.position) <= 3
        cond = (
            (target_cell.halite < target_cell.halite_local_mean * 0.75 ** 2)
            or (target_cell.halite < self.global_min_requirement_to_harvest_locally)
            or (
                    self.dist(spot, target_cell.position) <= 2
                    and self.is_pos_occupied_by_threat(ship, target_cell.position, is_equal_threat=False))
        )
        return cond

    def is_overharvested_globally(self, ship, target_cell):
        # Finished harvest?
        #   target_cell supplied as it may have been locally changed.
        #   POTENTIAL: ship.halite is more than saftey net  ship.halite > self.ship_carry_maximum or
        #   OR x4 local mean.
        cond = False if self.is_yard_emergancy else (
                ship.halite > self.ship_carry_maximum)
        # If ship carrying 2std than mean carrying, go home
        # gohome = ship.halite > self.padj_cargo_adj_mean + 2 * self.padj_cargo_adj_std
        return cond

    def determine_best_harvest_spot_while_depositing(self, ship):
        """Opportunistically harvest while moving back to yard.
        Spot above 40 min and harv increases cargo by >= 10%?"""
        cells = self.board.cells
        if cells[ship.position].halite > 40 and (cells[ship.position].halite * 0.25 / (ship.halite + 1)) > self.deposit_harvest_min_gain:
            return ship.position
        potential_pos2halite = {}
        for p in self.get_adjs(ship.position, r=1):
            if self.dist(ship.position, ship.log.yard.position) > self.dist(p, ship.log.yard.position):
                if cells[p].halite > 40 and (cells[p].halite * 0.25 / (ship.halite + 1)) > self.deposit_harvest_min_gain and p not in LOG.spots:
                    potential_pos2halite[p] = cells[p].halite
        if len(potential_pos2halite) > 0:
            return max(potential_pos2halite, key=potential_pos2halite.get)
        else:
            return None

    def get_closest_op_by_score(self):
        """Closest opponent by score."""
        deltas = {}
        mh = self.me.halite + sum([s.halite for s in self.me.ships])
        for pid in self.enemy_pids:
            p = self.board.players[pid]
            deltas[p] = abs(mh - p.halite + sum([s.halite for s in self.board.players[pid].ships]))
        co = min(deltas, key=deltas.get)
        return co

    def get_closest_opponents_by_score(self):
        """Closest opponents by score."""
        deltas = {}
        mh = self.me.halite + sum([s.halite for s in self.me.ships])
        for pid in self.enemy_pids:
            p = self.board.players[pid]
            deltas[p] = abs(mh - p.halite + sum([s.halite for s in self.board.players[pid].ships]))
        cos = sorted(deltas, key=deltas.get)
        return cos

    def get_attack_target(self, ship):
        cos = self.get_closest_opponents_by_score()
        valid_yards = []
        turns_left = self.board.configuration.episode_steps - self.board.step
        for co in cos:
            for sy in co.shipyards:
                dist = self.dist(sy.position, ship.position)
                if dist < turns_left:
                    valid_yards.append(sy)
        if len(valid_yards) > 0:
            target = valid_yards[0]
            return target.id
        else:
            return None

    def should_be_provident(self, ship):
        return False #self.count_adjacent_threats(ship, r=2, is_equal_threat=False) >= 3

    def assign_role(self, ship):
        """RETURN a new role"""
        if self.is_yard_emergancy:
            return 'emergancy'
        if ship.log.role == 'emergancy':
            # and not an emergancy?
            return 'DEP'
        if ship.log.role == 'NEW':
            return 'HVST'
        # Finished deposit? Return to harvest
        if ship.log.yard is not None:
            if ship.log.role == 'DEP' and ship.position == ship.log.yard.position:
                return 'HVST'
        if self.config.episode_steps - self.board.step <= 4 + self.dist(ship.position, ship.log.yard.position):
            adj_sys = [sy for sy in self.board.shipyards.values() if sy.player_id != self.me.id]
            if ship.halite > 0:
                return 'DEP'
            elif len(adj_sys) != 0 and len(self.me.ships) > 5:
                return 'strike_yards'
            else:
                return 'evade'
        am_in_lead = all([self.me.halite > net for p, net in self.p2net_halite.items() if p.id != self.me.id])
        if ship.log.role == 'HVST':
            if ship.log.is_frustrated:
                return 'DEP'
            if self.should_be_provident(ship):
                return 'DEP'
            temp_spot = ship.log.spot
            if temp_spot is None:
                temp_spot = self.determine_best_harvest_spot_from_yard(ship)
            if self.is_overharvested_locally(ship, temp_spot):
                # less than local stat? go somewhere else local
                temp_spot = self.determine_best_harvest_spot_locally(ship)
                if temp_spot is not None:
                    pass
                else:
                    return 'DEP'
            temp_spot = temp_spot if temp_spot is not None else ship.log.spot
            target_cell = self.board.cells[temp_spot]
            if self.is_overharvested_globally(ship, target_cell):
                return 'DEP'
        if ship.log.role == 'DFND':
            return 'DFND'
        if ship.log.role == 'assault':
            return 'assault'
        if ship.log.role == 'suppressor':
            return 'suppressor'
        return ship.log.role  # Should probably account for all cases above and catch anything here

    def evaluate_harvest_spot(self, ship):
        """Assign a new or reevaluate current harvest spot."""
        if ship.log.role in ('HVST', 'emergancy'):
            if ship.log.spot is None:
                return self.determine_best_harvest_spot_from_yard(ship)
            # **n harvests under local mean? try somewhere else OR if nearby threat on spot
            if self.is_overharvested_locally(ship, ship.log.spot):
                """less than local stat? go somewhere else local"""
                local_spot =  self.determine_best_harvest_spot_locally(ship)
                if local_spot is None:
                    return self.determine_best_harvest_spot_from_yard(ship)
                else: return local_spot
            else:
                return ship.log.spot
        elif ship.log.role in ('DEP',) and not self.should_be_provident(ship):
            return self.determine_best_harvest_spot_while_depositing(ship)
        else:
            return None  # Free up spot

    def get_best_ship_for_yard(self):
        """If building a yard after losing the only one:
            Return ship with minimum mean distance to others.
                Calculate distance between each point pair.
                Calculate mean of distance for each ships pairings.
        else if this will be 2nd yard:"""
        pair_dists = {}
        yardable_ships = [ship for ship in self.me.ships if ship.log.role in LOG.yardable_roles and (self.me.halite + ship.halite) >= 500]
        if len(self.me.shipyards) == 0:
            if len(self.me.ships) == 0:
                return None
            elif len(self.me.ships) == 1:
                return self.me.ships[0]
            else:   # Get most central ship
                ship_mean_dist = {}
                for pair in combinations([s.position for s in self.me.ships], 2):
                    pair_dists[pair] = self.dist(*pair)
                for ship in yardable_ships:
                    ship_mean_dist[ship] = np.mean([dist for pair, dist in pair_dists.items() if ship.position in pair])
                if len(ship_mean_dist) > 0:
                    return min(ship_mean_dist, key=ship_mean_dist.get)
                else:
                    return None
        if len(yardable_ships) == 0:
            return None
        else:
            # Get ship in appropriate spot
            sd = {}
            for s in yardable_ships:
                sydists = [self.dist(s.position, sy.position) for sy in self.me.shipyards]
                nearest_dist = self.dist(s.position, self.get_nearest_shipyard(s.position).position)
                not_adj_threats = all([not self.is_pos_occupied_by_threat(s, ppos) for ppos in self.get_adjs(s.position, r=1)])
                # Ensure ship in my quadrant if LT 5 yards
                is_in_quadrant = s.position in self.quadrant_points if len(self.me.shipyards) < 5 else True
                # Don't destroy a high value spot given that yard will probably be in your quadrant
                is_spot_low_halite = self.board.cells[s.position].halite < self.halite_global_mean * 0.75
                # Ship must be 6 spots away from all others but also within 8 from nearest.
                if is_spot_low_halite and is_in_quadrant and not_adj_threats and all([(6 <= dist) for dist in sydists]) and nearest_dist < 8:
                    # For ships within dist bounds, use ship near max density
                    sd[s] = self.halite_density[s.position]
            if len(sd) > 0:
                return max(sd, key=sd.get)
            else:
                return None

    def get_adjs(self, p, r=2, return_cells=False):

        pos_adjs = get_adjacent_positions(p, r)
        if return_cells:
            return [self.board.cells[p] for p in pos_adjs]
        else:
            return pos_adjs

    nships2nyards = {
        11:1,
        15:2,
        21:3,
        25:4,
        28:5,
    }

    def get_max_shipyards_by_nships(self):
        """How many yards to support as f(nships)"""
        nships = len(self.me.ships)
        if nships < 10:
            return 1
        elif nships < 15:
            return 2
        elif nships < 20:
            return 3
        elif nships < 30:
            return 4
        elif nships < 40:
            return 5
        else:
            return 6
        # return max((nships - 4) // 4, 1)

    def yard_converting(self):
        """Convert ships to yards.
        If no yards curretnly, create and mark prospective point"""
        self.is_yard_emergancy = len(self.me.shipyards) < 1
        max_shipyards = self.get_max_shipyards_by_nships()
        should_try_to_convert_yard = (self.board.step < magic['late_game_step']
                                      and len(self.me.shipyards) < max_shipyards
                                      and len(self.me.ships) > 10)
        cargosum = sum([s.halite for s in self.me.ships])
        late_game_and_worth_keeping_yard = cargosum > 1000 and self.board.step >= magic['late_game_step'] and len(self.me.shipyards) == 0
        early_game_and_no_yards = len(self.me.shipyards) == 0 and self.board.step < magic['late_game_step']
        if early_game_and_no_yards or late_game_and_worth_keeping_yard or should_try_to_convert_yard:
            ship = self.get_best_ship_for_yard()
            if ship is not None:
                ship.next_action = ShipAction.CONVERT
                ship.log.set_action = ShipAction.CONVERT
                # conversion is resolved before collision - we don't need to reserve point with log.set_point
                ship.log.p_point = None
                self.prospective_yard = Shipyard('PROSPECTIVE', ship.position, self.me.id, self.board)
                self.convertcount += 1
                self.is_yard_emergancy = False
            elif len(self.me.shipyards) == 0:  # Only an emergancy if zero yards
                self.prospective_yard = None
                self.is_yard_emergancy = True
        for s in self.me.ships:
            s.log.yard = self.get_nearest_shipyard(s.position)
            if self.is_yard_emergancy:
                s.log.role = 'emergancy'

    def get_adj_enemies(self, pos, radius=2, halite_min=9999):
        cells_adj = [self.board.cells[adj] for adj in self.get_adjs(pos, r=radius)]
        return [c.ship for c in cells_adj if c.ship is not None
                and c.ship.player_id != self.me.id
                and c.ship.halite < halite_min]

    def get_closest_enemy_ship(self, pos):
        if len(self.enemy_ships) > 0:
            ships2d = {ship: self.dist(pos, ship.position) for ship in self.enemy_ships}
            return min(ships2d, key=ships2d.get)
        else:
            return None

    def yard_defending(self, radius=3):
        """Assign new defender and evalute previous ones.
        No current defender? Try to assign:
            adjacent depositor
            spawn new ship
            closest ship
        Active defender? Try to re-assgin:
            adjacent depositor
            spawn new ship
        Don't assign an adjacent depositor if enemy threat also adjacent.
        """
        for sy in self.me.shipyards:
            enemy_adj = self.get_adj_enemies(sy.position, radius=radius)
            if len(enemy_adj) > 0 and sy not in LOG.yard2defenders:  # Try assign defender
                print(f'\n\toS:{enemy_adj[0].id}@{enemy_adj[0].position}')
                nearby_candidates_by_dist = [(s, self.dist(s.position, sy.position))
                                             for s in self.me.ships
                                             if self.dist(s.position, sy.position) <= radius
                                             and s.log.role != 'DFND']
                # try to assign an adj depositor
                adj_cells = [self.board.cells[pos] for pos in self.get_adjs(sy.position, r=1)]
                adj_deprs = [c.ship for c in adj_cells if
                             c.ship is not None and c.ship.player_id == self.me.id and c.ship.log.role == 'DEP']
                adj_threats = [c.ship for c in adj_cells if
                             c.ship is not None and c.ship.player_id != self.me.id and c.ship.halite == 0]

                if len(adj_deprs) > 0 and len(adj_threats) == 0:
                    bship = max(adj_deprs, key=lambda x: x.halite)
                    bship.log.track_id = sy.id
                    bship.log.role = 'DFND'
                elif (self.should_still_spawn) and self.can_afford_to_spawn():
                    sy.next_action = ShipyardAction.SPAWN
                    self.spawncount += 1
                elif len(nearby_candidates_by_dist) > 0:
                    ship = min(nearby_candidates_by_dist, key=lambda x: x[1])[0]
                    print(f'\tdfnd:{ship.id}@{ship.position}\n')
                    ship.log.track_id = sy.id
                    ship.log.role = 'DFND'
            elif sy in LOG.yard2defenders:  # Re-evaluate
                # If immediately safe go back to harvesting
                ship = LOG.yard2defenders[sy]
                if len(self.get_adj_enemies(ship.log.yard.position, radius=radius)) == 0:
                    print(f"\n\tRelieving defender: {ship.id} at P{ship.position}\n")
                    ship.log.track_id = None
                    ship.log.role = 'HVST'
                elif ship.position == sy.position:
                    # try to assign an adj depositor
                    adj_cells = [self.board.cells[pos] for pos in self.get_adjs(sy.position,r=1)]
                    adj_deprs = [c.ship for c in adj_cells if c.ship is not None and c.ship.player_id == self.me.id and c.ship.halite > 0]
                    adj_threats = [c.ship for c in adj_cells if
                                   c.ship is not None and c.ship.player_id != self.me.id and c.ship.halite == 0]

                    if len(adj_deprs) > 0 and len(adj_threats) == 0:
                        bship = max(adj_deprs, key=lambda x:x.halite)
                        bship.log.track_id = sy.id
                        bship.log.role = 'DFND'
                        ship.log.track_id = None
                        ship.log.role = 'HVST'
                    else:
                        if self.should_still_spawn and self.can_afford_to_spawn():
                            sy.next_action = ShipyardAction.SPAWN
                            self.spawncount += 1
                            ship.log.track_id = None
                            ship.log.role = 'HVST'
                else:
                    # try to assign any closer ships
                    nearby_candidates_by_dist = [(s, self.dist(s.position, sy.position))
                                                 for s in self.me.ships
                                                 if self.dist(s.position, sy.position) <= radius
                                                 and s.log.role != 'DFND']
                    if len(nearby_candidates_by_dist) > 0:
                        closest_ship = min(nearby_candidates_by_dist, key=lambda x:x[1])[0]
                        if closest_ship != ship:
                            closest_ship.log.track_id = sy.id
                            closest_ship.log.role = 'DFND'
                            ship.log.track_id = None
                            ship.log.role = 'HVST'

    def assign_assaulter(self, ship, esy):
        """Update ship log for assualting."""
        print(f"\tAssigning {ship.id}@{ship.position}\tto ASSAULT {esy.id}@{esy.position}")
        ship.log.track_id = esy.id
        ship.log.role = 'assault'

    def yard_assaulting(self):
        """Assign one ship to assault yards near my yards"""
        candidate_esys = [esy for esy in self.board.shipyards.values()
                          if esy.player_id != self.me.id
                          and self.dist(esy.position, self.get_nearest_shipyard(esy.position).position) <= self.yard_assualt_min_radius]
        for esy in candidate_esys:
            if esy not in LOG.yard2assaulters:
                possible_assaulters = [s for s in self.me.ships if s.halite == 0 and s.log.role not in ('assault', 'DFND')]
                if len(possible_assaulters) > 0:
                    best_assaulter = min(possible_assaulters, key=lambda s:get_dist(s.position, esy.position))
                    self.assign_assaulter(best_assaulter, esy)

    def assign_suppressors(self):
        """Estimate how many ships can take on conflict roles and assigning here."""
        n_suppressors = len(LOG.get_suppressors())
        max_suppressors = len(self.me.ships) - len([s for s, w in self.harvest_spot_values if w > self.spot_weight_threshold])/4
        print(f"Max Suppressors:{max_suppressors}")

        if max_suppressors > n_suppressors:
            ships_zero = [s for s in self.me.ships if s.log.role in LOG.expendable_roles and s.halite == 0]
            ships_notzero = [s for s in self.me.ships if s.log.role in LOG.expendable_roles and s.halite > 0]
            ships_priority = ships_zero + sorted(ships_notzero, key=lambda x:-self.dist(x.position, x.log.yard.position))
            while (len(LOG.get_suppressors()) < max_suppressors) and len(ships_priority) > 0:
                ship = ships_priority.pop()
                ship.log.role = 'suppressor'
        elif max_suppressors < n_suppressors:
            # Reassign suppressors closest to yard as harvestors
            ships_priority = sorted(LOG.get_suppressors(), key=lambda x: -self.dist(x.position, x.log.yard.position))
            while (len(LOG.get_suppressors()) > max_suppressors) and len(ships_priority) > 0:
                ship = ships_priority.pop()
                ship.log.role = 'HVST'

    def assemble_squadrons(self):
        """
            Initiate a new squadron if there is capacity
            Remove destoryed ids from squad
            Assign suppressors to squads
            Connect squad to ship logEntry
            Refresh target for squadrons
        """
        nsq = len(LOG.squadrons)
        mnsq = self.maximum_squadrons()
        c = count(0, 1)
        if nsq < mnsq:
            # generate one squadron per turn
            sqid = f"{self.board.step}-{next(c)}"
            sq = Squadron(sqid)
            LOG.squadrons.append(sq)
        for sq in LOG.squadrons:
            sq.refresh_entities()
        for ship in LOG.get_suppressors():  # If ship not already assigned to a squad, assign
            if ship.id not in LOG.get_squaddie_ids:
                sq = min(LOG.squadrons, key=lambda sq: sq.nships)
                sq.assign(ship)
        for sq in LOG.squadrons:  # Clear empty squadrons, else do some setup
            if sq.nships > 0:
                sq.attach_squad_to_ships()
                sq.update_midpoint()
                sq.refresh_targets()
            else:
                LOG.squadrons.remove(sq)
                del sq

    def get_density_maxima(self):
        """Get a set of positions where the halite density appears to be a maxima.
        Return positions where the adjacent halite density is equal or decreasing.
        """
        dmax = {}
        for p,v in self.halite_density.items():
            adjs = self.get_adjs(p, r=1)
            conds = [self.halite_density[adj] <= v for adj in adjs]
            if all(conds):
                dmax[p] = v
        return dmax

    def get_suppress_value(self, ship):
        """Rating of enemy ships suppress worth - Not used."""
        if ship.halite == 0:
            return -1
        else:
            value = (500-ship.halite)/(1 + self.dist(ship.position, self.get_nearest_shipyard(ship.position).position))**2
            return value

    def get_endgame_spot_weight_threshold(self):
        """Approaching the end game, lower the threashold for deciding if a spot is worth harvesting."""
        steps_left = self.board.configuration.episode_steps - self.board.step
        return 5 * (steps_left - 80)/(150 - 80)

    def setup_stats(self):
        """Computables at start of each step."""
        setupt0 = time.time()
        LOG.enemy_targ_ids = [s.id for s in self.board.ships.values() if s.player_id != self.me.id] + [sy.id for sy in self.board.shipyards.values() if sy.player_id != self.me.id]
        self.am_in_halite_lead = all(
            [(self.me.halite > p.halite) for p in self.board.players.values() if p.id != self.me.id])
        self.enemy_shipyards = [sy for sy in self.board.shipyards.values() if sy.player_id != self.me.id]
        self.enemy_ships = [s for s in self.board.ships.values() if s.player_id != self.me.id]
        self.suppress_value = {s: self.get_suppress_value(s) for s in self.enemy_ships}
        self.suppress_value = OrderedDict(sorted(self.suppress_value.items(), key=lambda x:x[1], reverse=True))
        self.suppress_value_rank = {sid: rank for rank, sid in enumerate(self.suppress_value)}
        # Map halite
        cells_with_halite = [cell.halite for cell in self.board.cells.values() if cell.halite != 0]
        self.halite_global_mean = int(np.mean(cells_with_halite))
        self.halite_global_median = int(np.median(cells_with_halite))
        self.halite_global_std = int(np.std(cells_with_halite))
        cells = self.board.cells
        cos = self.get_closest_opponents_by_score()
        self.best_suppress_yards = []
        for o in cos:
            if len(o.shipyards) > 0:
                self.best_suppress_yards = o.shipyards
                break
        self.ship_and_yard_density = {}
        for p, cell in cells.items():
            adjs = self.get_adjs(p, r=3)
            if cell.halite > 0:
                halites = [(cells[ap].halite / self.dist(p, ap)) for ap in adjs] + [cell.halite]
            else:
                halites = [0]
            if cell.ship is not None and cell.ship.player_id != self.me.id:
                enemy_cargos = [((cells[ap].ship.halite / self.dist(p, ap))  # Yikes!
                                 if cells[ap].ship is not None and cells[ap].ship.player_id != self.me.id else 0)
                                for ap in adjs]\
                               + [cell.ship.halite if cell.ship is not None and cell.ship.player_id != self.me.id else 0]
            else:
                enemy_cargos = [0]
            self.halite_density[p] = np.mean(halites)
            # self.suppress_value_density[p] = np.mean(suppress_values)
            self.cargo_density[p] = np.mean(enemy_cargos)
            # Need large radius for collisionables to effectively nudge movement
            collisionable_count = [(cells[ap].ship is not None or cells[ap].shipyard is not None)
                                   for ap in self.get_adjs(p, r=2)]
            self.ship_and_yard_density[p] = sum(collisionable_count)/len(collisionable_count)
            self.halite_density[p] = np.mean([cells[ap].halite for ap in self.get_adjs(p, r=4)] + [cell.halite])
            halites = [cells[ap].halite for ap in self.get_adjs(p, r=3) if cells[ap].halite != 0] + [cell.halite]
            halites = halites if len(halites) > 0 else [0]
            # Local stats do not include 0 halite cells
            cell.halite_local_mean = np.mean(halites)
            cell.halite_local_std = np.std(halites)
        self.max_cargo_density_pos = max(self.cargo_density, key=self.cargo_density.get)
        print(f'Max Cargo density point@{self.max_cargo_density_pos}')
        # self.max_suppress_value_density_pos = max(self.suppress_value_density, key=self.suppress_value_density.get)
        self.halite_density_maxima = self.get_density_maxima()
        # Cargo halite
        adjop_cargos = [s.halite for s in self.board.ships.values() if s.player_id in self.pids_adj]
        adjop_cargos = [0,] if len(adjop_cargos) == 0 else adjop_cargos
        me_cargos = [s.halite for s in self.me.ships]
        me_cargos = [0, ] if len(me_cargos) == 0 else me_cargos
        self.padj_cargo_adj_mean = np.mean(adjop_cargos)
        self.padj_cargo_adj_std = np.std(adjop_cargos)
        self.me_cargo_mean = np.mean(me_cargos)
        self.me_cargo_std = np.std(me_cargos)
        for pid, p in self.board.players.items():
            self.p2net_halite[p] = p.halite + sum([s.halite for s in self.board.ships.values()])

        # Calculate ratings for potential harvest spots.
        adj_eyard_spots = []
        for sy in self.enemy_shipyards:
            adj_eyard_spots.extend([adj for adj in self.get_adjs(sy.position, r=1)])
        if self.yardcount > 0 or self.prospective_yard is not None:
            d = OrderedDict()
            pos_yards = {x.cell.position for x in self.board.shipyards.values()}
            est_harvest_time = 3
            for pos, cell in self.board.cells.items():
                if pos not in adj_eyard_spots:
                    # Assumes first SY is on first ship init point. This needs to change if first SY can change location
                    sy_pos = self.get_nearest_shipyard(pos).position
                    # Nearby spots are safest - don't need to look overly enticing just because they're close
                    dist_actual = self.dist(pos, sy_pos)
                    dist_penalty = max(dist_actual, 2)
                    halite_expected = min(cell.halite * (1 + self.config.regen_rate) ** dist_actual,
                                          self.config.max_cell_halite)  # Potential on arrival
                    halite_harvest = halite_expected * (1 - 0.75 ** est_harvest_time)  # Model expected halite after mining?
                    halite_potential = halite_harvest / (2 * dist_penalty + est_harvest_time)  # There and back again...
                    d[pos] = halite_potential
                else:
                    d[pos] = -1
            self.harvest_spot_values = sorted(d.items(), key=lambda x: -x[1])

        # Calculate per turn constants
        # self.halite_harvest_minimum = self.halite_global_mean - self.halite_global_std
        # self.global_min_requirement_to_harvest = self.halite_global_mean - self.halite_global_std
        self.global_min_requirement_to_harvest = 40 if self.board.step < 350 else 20
        self.global_min_requirement_to_harvest_locally = 40 if self.board.step < 350 else 20
        if self.board.step < magic['early_game_step']:  # Greedy
            self.spot_weight_threshold = 0
            self.deposit_harvest_min_gain = 0.10
            self.ship_carry_maximum = 300
            self.frustration_max = 2
        elif self.board.step < magic['late_game_step']:  # Speedy
            self.spot_weight_threshold = 5
            self.deposit_harvest_min_gain = 0.10
            self.ship_carry_maximum = 300
            self.frustration_max = 2
        else:  # Needy
            self.spot_weight_threshold = self.get_endgame_spot_weight_threshold()
            self.deposit_harvest_min_gain = 0.10
            self.ship_carry_maximum = 300
            self.frustration_max = 2
        self.yard_assualt_min_radius = 4
        self.enemy_ship_points = [ship.position for plr in self.board.players.values()
                                  if plr is not self.me for ship in plr.ships]
        self.enemy_yards = [sy for sy in self.board.shipyards.values() if sy.player_id != self.me.id]
        print(f'Time taken for setup():{time.time() - setupt0}')

    def eval_frustration(self, ship):
        """Ships track frustration across steps, sending them home if they are forced away from their goal."""
        if ship.log.target_cell is not None:  # Else its converting or possibly some other cases.
            best_actions = self.get_best2worst_actions(ship, ship.log.target_cell, weighted=False)
            best_action = best_actions[0]
            pos_adjs = [ship.position.translate(adelta[a], self.dim) for a in best_actions if a not in (None,)]
            if any([self.is_pos_occupied_by_threat(ship, pos_adj) for pos_adj in pos_adjs]):
                ship.log.frustration += 1
            elif ship.log.set_action != best_action and ship.log.set_action == 'WAIT':
                ship.log.frustration += 1
            else:  # reduce frustration, back to zero.
                ship.log.frustration = max(ship.log.frustration - 1, 0)
            # If ship is not waiting about and is on a shipyard, reset frustration
            if ship.log.set_action != 'WAIT':
                ship.log.frustration = 0 if self.board.cells[ship.position].shipyard is not None else ship.log.frustration
            current_cell = self.board.cells[ship.position]
            if ship.next_action != ShipAction.CONVERT:
                next_cell = self.board.cells[ship.position.translate(adelta[ship.log.set_action], self.dim)]
                if next_cell.shipyard is not None or current_cell.shipyard is not None:
                    # Reset frustration now if next turn is on shipyard. Current cell used as redundancy.
                    ship.log.frustration = 0
            ship.log.is_frustrated = ship.log.frustration >= self.frustration_max

    def ship_spawning(self):
        """Sets a boolean attribute if we should try to spawn any ships this turn."""
        me = self.me
        # SHIP BUILDING
        # Match other players max ship count
        p2ns = [(p.id, len(p.ships)) for p in self.board.players.values()]
        p2ns_notme = sorted([(p,ns) for p,ns in p2ns if p != self.me.id], key=lambda x:-x[1])
        sorted_other_net_halite = sorted( (
            (p,(p.halite + sum([s.halite for s in self.board.players[p.id].ships])))
            for p in self.board.players.values() if p.id is not me.id), key=lambda x:x[1], reverse=True)
        max_other_net_halite = sorted_other_net_halite[0][1]
        second_other_net_halite = sorted_other_net_halite[1][1]
        # Spawn in early game, or maintain 5 in endgame
        max_p2ns = max(p2ns, key=lambda x: x[1])
        max_ns_other_pid, max_ns_other = max([(pid,ns) for pid, ns in p2ns if pid != me.id], key=lambda x: x[1])
        min_ns_other_pid, min_ns_other = min([(pid, ns) for pid, ns in p2ns if pid != me.id], key=lambda x: x[1])
        max_nships_everyone = max_p2ns[1]
        my_net_halite = me.halite + sum([s.halite for s in self.me.ships])
        delta_max_halite = my_net_halite - max_other_net_halite
        delta_second_halite = my_net_halite - second_other_net_halite
        delta_min_halite = me.halite - self.board.players[min_ns_other_pid].halite
        delta_max_ships = len(me.ships) - max_ns_other
        delta_min_ships = len(me.ships) - min_ns_other
        # If ahead on net halite, and not more than 5 ships above next highest, spawn another
        am_maintaining_lead = max_p2ns[0] == me.id and delta_max_halite >= 500
        am_lead_on_second = delta_second_halite >= 500
        am_below_min_op = len(me.ships) < min_ns_other and delta_min_halite >= 500
        if self.board.step < 100:
            should_still_spawn = True
        elif 100 <= self.board.step < magic['late_game_step']:
            should_still_spawn = am_lead_on_second or len(self.me.ships) < p2ns_notme[1][1]
        else :  # (obs.step >= magic['late_game_step']) naturally True
            should_still_spawn = len(me.ships) < 5
        self.should_still_spawn = should_still_spawn

    def get_shipsets_by_priority_threat(self):
        """Group ships by priority due to adjacent threats."""
        priority2ships = {}
        for ship in self.me.ships:
            if ship.log.adj_threats not in priority2ships:
                priority2ships[ship.log.adj_threats] = [ship]
            else:
                priority2ships[ship.log.adj_threats].append(ship)
        return [ships for priority, ships in sorted(priority2ships.items(), key=lambda x:x[0], reverse=True)]

    def get_shipsets_by_priority_allies(self, ships):
        """Group ships by priority due to adjacent Allies.
        Ships in the centre of a cluster should go first as their choices are more limited."""
        priority2ships = {}
        for ship in ships:
            if ship.log.adj_allies not in priority2ships:
                priority2ships[ship.log.adj_allies] = [ship]
            else:
                priority2ships[ship.log.adj_allies].append(ship)
        return [ships for priority, ships in sorted(priority2ships.items(), key=lambda x:x[0], reverse=True)]

    """                     GET ACTIONS                 """
    def get_actions(self, obs, config):
        """Main loop"""
        self.board = Board(obs, config)
        self.me = self.board.current_player
        me = self.me  # just for shorthand
        self.spawncount, self.convertcount, self.yardcount = 0, 0, len(self.me.shipyards)
        self.refresh_log()
        self.yard_converting()
        self.ship_spawning()
        self.setup_stats()
        self.yard_defending()
        self.yard_assaulting()
        self.assign_suppressors()
        self.assemble_squadrons()
        self.action_iter = 0

        # Initial evaluation loop
        for ship in [s for s in me.ships if s.log.set_action is None]:
            ship.log.role = self.assign_role(ship)
            ship.log.spot = self.evaluate_harvest_spot(ship)
            ship.log.target_cell = self.determine_target_cell(ship)

        # Resolve ship actions in nested sets, grouped by number of adjacent threats then adjacent allies
        # Lets ships with less options go first.
        ships_by_threat = self.get_shipsets_by_priority_threat()
        for shipset_threat in ships_by_threat:
            ships_by_allies = self.get_shipsets_by_priority_allies(shipset_threat)
            for shipset_active in ships_by_allies:
                # Main ship loop - iterate until each ship has an action
                while any([s.log.set_action is None for s in shipset_active]):
                    self.action_iter += 1
                    if self.action_iter > 24:
                        raise BaseException(f"action resolution iteration > 24 - probable infinite loop")

                    # Calculate best potential actions
                    for ship in [s for s in shipset_active if s.log.set_action is None]:
                        ship.log.p_action = self.move_to_target(ship, ship.log.target_cell)
                        if ship.log.p_action != ShipAction.CONVERT:  # Will only convert if there are no safe moves.
                            ship.log.p_point = ship.position.translate(adelta[ship.log.p_action], self.dim)
                        else:
                            ship.log.set_action = ShipAction.CONVERT
                            ship.next_action = ShipAction.CONVERT
                            self.convertcount += 1

                    # ACTION CONFLICT RESOLUTION
                    # Confirm non-conflicting actions. Record set actions in ship log to keep track of
                    # how many ships actions are finalized. Resolve actions outwards from shipyards.
                    p2ships = LOG.point2potential_ships(shipset_active)
                    for point in p2ships:
                        ships = p2ships[point]
                        if len(ships) == 1:  # Only contender - give action
                            ship = ships[0]
                            action, point = ship.log.p_action, ship.log.p_point  # any set points should now NOT be possibilities.
                        else:  # Give spot to highest priority ship
                            dfnd_ships = [s for s in ships if s.log.role == 'DFND']
                            sy_ships = [s for s in ships if s.position in [sy.position for sy in me.shipyards]]
                            ships_by_threats_then_halite = sorted(ships, key=lambda x: (x.log.adj_threats, x.halite), reverse=True)
                            ships_by_priority = dfnd_ships + sy_ships + ships_by_threats_then_halite
                            ship = ships_by_priority[0]
                            action, point = ship.log.p_action, ship.log.p_point
                        ship.next_action = action if action != 'WAIT' else None
                        ship.log.set_action, ship.log.set_point = action, point

        for shipyard in me.shipyards:
            # If we can afford spawn, considering cumulation of other SY spawns and keeping a reserve for one yard.
            if shipyard.next_action is None:  # Defender stage may have already set spawn command.
                no_ship_reserved_point = shipyard.position not in LOG.set_points
                if self.can_afford_to_spawn() and self.should_still_spawn and no_ship_reserved_point:
                    shipyard.next_action = ShipyardAction.SPAWN
                    self.spawncount += 1
        for ship in me.ships:
            self.eval_frustration(ship)
            ship.log.last_action = ship.log.set_action
        return me.next_actions


def agent(obs, config):
    global myBot, myId
    if obs.step == 0:
        precompute()
    if 'myBot' not in globals():
        myBot = {obs.player: MyAgent(obs, config)}
        myId = obs.player
    actions = myBot[obs.player].get_actions(obs, config)
    return actions
