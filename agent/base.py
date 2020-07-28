from kaggle_environments.envs.halite.helpers import Board, ShipAction, ShipyardAction, Ship, Shipyard, Point
from collections import OrderedDict
from functools import lru_cache
from itertools import combinations, product
import numpy as np

np.seterr(all='raise')
"""NEXT
    -live
        get replayer working
        turn 30 why out of position?  
        turn 15 - ship deposits but returns to same spot. go home needs to be more liberal/ dependant on threat.

    -improve harvest logic
        if spot.halite < threshold, go to nearby spot Why is ship returning home so early?

    -improve next move logic
        - consider halite in cells incase ship ends up waiting
        - consider desirable collisions

    -endgame
        call home
        swarm top player

    -build extra yard at some point
    if step > 20 and halite > reserve + shipspawn + yardspawn and nyards < 2:
        build yard at midpoint of most halite dense spot and yard 1

    -shipbuilding
        if nstep < 300 and totalHalite/4 > 2000

    -each ship should have its own threat modifier for cells
        -ve if ship.halite < avg. enemy
        +ve if ship.halite > avg. enemy

    -advanced pathing to avoid threat dense areas

    -assault role, enabled when not worth mining
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

pid2quadrant = {
    0: Point(5, 15),
    1: Point(15, 15),
    2: Point(5, 5),
    3: Point(15, 5)
}


class LogEntry:
    """Like a dictionary, without the ['fluff']"""

    def __init__(self):
        self.role = 'NEW'
        self.spot = 'NEW'
        self.spot_local = 'NEW'
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
        self.board = Board(obs, config)
        self.board_prev = None
        self.config = self.board.configuration
        self.me = self.board.current_player
        self.dim = config.size
        self.mid = config.size // 2
        self.quadrant_position = pid2quadrant[self.me.id]
        self.quadrant_points = self.get_quadrant_points()
        self.quadrant_adjs = set(self.get_adjs(self.quadrant_position, r=1))
        self.harvest_spot_values = None
        self.enemy_ship_points = None
        self.halite_global_mean = None
        self.halite_global_median = None
        self.halite_global_std = None
        self.halite_density = None
        self.yardcount = None
        self.prospective_yard = None
        self.action_iter = None
        self.keep_spawning_tripswitch = True
        self.cell_halite_minimum = None
        self.ship_carry_maximum = None  # calculated at start of each loop
        self.halite_harvest_minimum = None
        self.generate_constants()

    def get_quadrant_points(self):
        """Define player quadrant incl. shared points."""
        points = []
        qp = (self.mid // 2) + 1
        for x in range(-qp, qp + 1):
            for y in range(-qp, qp + 1):
                points.append(self.quadrant_position + (x, y))
        return points

    def generate_constants(self):
        self.cell_halite_minimum = 0

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

    @lru_cache(maxsize=21 ** 2)
    def dist(self, p1, p2):
        p1, p2 = np.array(p1), np.array(p2)
        p = abs(p1 - p2)
        y, x = min(p[0], self.dim - p[0]), min(p[1], self.dim - p[1])
        return abs(sum([x, y]))

    def get_nearest_shipyard(self, pos):
        if self.yardcount == 0:
            return self.prospective_yard
        if self.yardcount == 1:
            return self.me.shipyards[0]
        else:
            yards = [(self.dist(pos, sy.position), sy) for sy in self.me.shipyards]
            yards = sorted(yards, key=lambda x: x[0])
            return yards[0][1]

    @staticmethod
    def assign_role(ship):
        ship.log.role = 'HVST'
        ship.log.spot = None

    def determine_best_harvest_spot_locally(self, ship):
        d = {}
        for pos in self.get_adjs(ship.position, r=2):
            d[pos] = self.board.cells[pos].halite
        d = sorted(d, key=d.get)[::-1]
        for pos in d:
            if pos not in LOG.set_points and not self.is_pos_occupied_by_threat(ship, pos):
                return pos
        return ship.position

    def determine_best_harvest_spot_from_yard(self, ship):
        # Choose a spot to harvest - values already sorted desceding.
        # TODO - harvest_spot_values should be weighted by distance to midpont of ship and nearestSY
        # TODO - use zone to determine initial spot, then local values
        # def gen_local_harvest_spot_values():
        #     mp = get_mid_point(ship, ship.nearestSY)
        #     weights = {}
        #     for point in pointsWithinRadiusOfShip:
        #         weights{point} = point.halite / dist(mp, point)  # TODO how to weight distance?
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
        n_conditions = 4
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
                is_not_blocking_yard = (
                            ship.log.role != 'DEP' and ppos not in [sy.position for sy in self.me.shipyards])
                is_not_occupied_by_potential_threats = all(
                    [not self.is_pos_occupied_by_threat(ship, ppos_adj) for ppos_adj in ppos_adjs])
                # Conditions are ordered by priority
                conditions = [is_not_occupied_by_threat, is_not_occupied_by_self, is_not_blocking_yard,
                              is_not_occupied_by_potential_threats]
                if all(conditions[0:n_conditions]):
                    chosen_action = action
                    break
            n_conditions -= 1
            if n_conditions == 0:
                chosen_action = ShipAction.CONVERT  # No good moves found
                # TODO log this - Might need to enforce .next_action here and bypass action resolve cycle.
                break
        return chosen_action

    def determine_best_harvest_action(self, ship):
        if not ship.position == ship.log.spot_local:
            ship.log.p_action = self.move_to_target(ship, ship.log.spot_local)
            if ship.log.p_action != ShipAction.CONVERT:  # Will only convert if there are no safe moves.
                ship.log.p_point = ship.position.translate(adelta[ship.log.p_action], self.dim)
            else:
                ship.log.set_action = ShipAction.CONVERT
                ship.next_action = ShipAction.CONVERT
        else:  # harvest - identical for now!
            ship.log.p_action = self.move_to_target(ship, ship.log.spot_local)
            if ship.log.p_action != ShipAction.CONVERT:  # Will only convert if there are no safe moves.
                ship.log.p_point = ship.position.translate(adelta[ship.log.p_action], self.dim)
            else:
                ship.log.set_action = ShipAction.CONVERT
                ship.next_action = ShipAction.CONVERT

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
            if ship.log.spot is None:
                ship.log.spot = self.determine_best_harvest_spot_from_yard(ship)
                ship.log.spot_local = ship.log.spot
            target_cell = self.board.cells[ship.log.spot_local]
            # **n harvests under local mean? try somewhere else
            if target_cell.halite < target_cell.halite_local_mean * 0.75 ** 1:
                """less that local stat? go somewhere else local"""
                ship.log.spot = self.determine_best_harvest_spot_locally(ship)
                ship.log.spot_local = ship.log.spot
                target_cell = self.board.cells[ship.log.spot_local]
            if self.dist(ship.position, target_cell.position) <= 2 and self.is_pos_occupied_by_threat(ship,
                                                                                                      target_cell.position):
                ship.log.spot = self.determine_best_harvest_spot_locally(ship)
                ship.log.spot_local = ship.log.spot
                target_cell = self.board.cells[ship.log.spot_local]
            # Finished harvest?
            #   POTENTIAL: ship.halite is more than saftey net  ship.halite > self.ship_carry_maximum or
            #   OR x4 local mean.
            if ship.halite > (target_cell.halite_local_mean * 4):
                ship.log.role = 'DEP'
            else:
                self.determine_best_harvest_action(ship)
        if ship.log.role == 'DEP':
            self.determine_best_deposit_action(ship)

    def get_best_ship_for_yard(self):
        """If building a yard after losing the only one:
            Return ship with minimum mean distance to others.
                Calculate distance between each point pair.
                Calculate mean of distance for each ships pairings.
        else if this will be 2nd yard:
            # TODO - simple logic. need to perhaps consider halite density and send a ship to a pos to build it"""
        pair_dists = {}
        if len(self.me.ships) == 1:
            return self.me.ships[0]
        elif len(self.me.shipyards) == 0:
            ship_mean_dist = {}
            for pair in combinations([s.position for s in self.me.ships], 2):
                pair_dists[pair] = self.dist(*pair)
            for ship in self.me.ships:
                ship_mean_dist[ship] = np.mean([dist for pair, dist in pair_dists.items() if ship.position in pair])
            return sorted(ship_mean_dist, key=ship_mean_dist.get)[0]
        else:
            sd = {}
            for s in self.me.ships:
                dist = self.dist(s.position, self.me.shipyards[0].position)
                # hard cody mc hardcodeface
                if dist > 3 and dist < 10 and self.board.cells[
                    s.position].halite < self.halite_global_mean * 0.75 and s.position in self.quadrant_points:
                    sd[s] = self.dist(s.position, self.me.shipyards[0].position)  # working case assuming only 1 SY
            if len(sd) > 0:
                return sorted(sd, key=sd.get)[::-1][0]
            else:
                return None
            # ph = {}
            # for pid, p in self.board.players.items():
            #     if p.id != self.me.id:
            #         ph[pid] = p.halite + sum([s.halite for s in self.board.ships.values() if s.id == pid])
            # lowest_ph_pids =  sorted(ph, key=ph.get)[0:2]
            # pos = pid2quadrant[lowest_ph_pids]

    @lru_cache(maxsize=21 ** 2)
    def get_adjs(self, p, r=2):
        coords = [x for x in range(-r, r + 1)]
        # Get product of coords where sum of abs values is <= radius of average area
        # Mod to map coord space
        adjs = [x for x in product(coords, coords) if sum([abs(c) for c in x]) <= r]
        adjs.remove((0, 0))
        pos_adjs = [((p + x) % self.dim) for x in adjs]
        return pos_adjs

    def start_of_turn_yard_spawning(self):
        # If no yards, create and mark point
        if len(self.me.shipyards) == 0:
            ship = self.get_best_ship_for_yard()
            ship.next_action = ShipAction.CONVERT
            ship.log.set_action = ShipAction.CONVERT
            # conversion is resolved before collision - we don't need to reserve point with log.set_point
            ship.log.p_point = None
            self.prospective_yard = Shipyard('PROSPECTIVE', ship.position, self.me.id, self.board)
        # Quick try for second SY logic
        if self.board.step > 20 and self.me.halite > 2000 and len(self.me.ships) > 10 and len(self.me.shipyards) == 1:
            ship = self.get_best_ship_for_yard()
            if ship is not None:
                ship.next_action = ShipAction.CONVERT
                ship.log.set_action = ShipAction.CONVERT
                ship.log.p_point = None

    def setup_stats(self):
        """Computables at start of each step. Lazily calculating adjacent positions for each position."""
        self.halite_global_mean = int(np.mean([cell.halite for cell in self.board.cells.values() if cell.halite != 0]))
        self.halite_global_median = int(
            np.median([cell.halite for cell in self.board.cells.values() if cell.halite != 0]))
        self.halite_global_std = int(np.std([cell.halite for cell in self.board.cells.values() if cell.halite != 0]))
        # g = np.ndarray([self.dim, self.dim])
        self.halite_density = {}
        cells = self.board.cells
        for p, cell in cells.items():
            # g[p] = cell.halite
            self.halite_density[p] = np.mean([cells[ap].halite for ap in self.get_adjs(p, r=2)] + [cell.halite])
            halites = [cells[ap].halite for ap in self.get_adjs(p, r=2) if cells[ap].halite != 0] + [cell.halite]
            halites = halites if len(halites) > 0 else [0]
            cell.halite_local_mean = round(np.mean(halites), 1)
            cell.halite_local_std = round(np.std(halites), 1)

        # Calculate ratings for potential harvest spots.
        if self.yardcount > 0:
            d = OrderedDict()
            pos_yards = {x.cell.position for x in self.board.shipyards.values()}
            est_harvest_time = 5
            for pos, cell in self.board.cells.items():
                # Exclude points outside of defined quadrant, yards, and the points immediate to the first shipyard
                # TODO assumes first SY is on first ship init point. This needs to change if first SY can change location
                if pos in (set(self.quadrant_points) - pos_yards - self.quadrant_adjs):
                    sy_pos = self.get_nearest_shipyard(pos).position
                    dist = self.dist(pos, sy_pos)
                    halite_expected = min(cell.halite * 1 + self.config.regen_rate ** dist,
                                          self.config.max_cell_halite)  # Potential on arrival
                    halite_harvest = halite_expected * (1 - 0.75 ** est_harvest_time)  # Model expected halite after mining?
                    halite_potential = halite_harvest / (2 * dist + est_harvest_time)  # There and back again...
                else:
                    halite_potential = -1
                d[pos] = halite_potential
            # get spots around enemy yards to ignore
            enemy_yards_pos = [yard.position for yard in self.board.shipyards.values()
                               if yard.player_id != self.me.id]
            # spots_ignore = set([pos for posyard in enemy_yards_pos for pos in self.get_adjs(posyard, r=3)] + enemy_yards_pos)
            self.harvest_spot_values = sorted(d.items(), key=lambda x: -x[1])

        # Calculate per turn constants
        self.halite_harvest_minimum = self.halite_global_mean - self.halite_global_std
        self.ship_carry_maximum = self.halite_global_mean + 3 * self.halite_global_std
        self.enemy_ship_points = [ship.position for plr in self.board.players.values()
                                  if plr is not self.me for ship in plr.ships]

    def get_actions(self, obs, config):
        """Main loop"""
        self.board = Board(obs, config)
        self.me = self.board.current_player
        me = self.me  # just for shorthand
        spawncount = 0
        self.refresh_ships()
        self.yardcount = len(self.me.shipyards)
        self.setup_stats()
        self.start_of_turn_yard_spawning()
        # Main ship loop - iterate until each ship has an action
        self.action_iter = 0
        while len(LOG.set_actions) != len(me.ships):
            self.action_iter += 1
            if self.action_iter > 24:
                raise BaseException(f"action resolution iteration > 24 - probable infinite loop")
            if self.action_iter % 10 == 0:
                print(f"Action Iter:{self.action_iter}")

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
                else:  # Give spot to highest priority ship (first ship found on a yard else ship with highest halite)
                    syships = [s for s in ships if s.position in [sy.position for sy in me.shipyards]]
                    if len(syships) > 0:
                        priority_ship = syships[0]
                    else:
                        ships_by_halite = sorted([(s, s.halite) for s in ships], key=lambda x: -x[1])
                        priority_ship, halite = ships_by_halite[0]
                    action, point = priority_ship.log.p_action, priority_ship.log.p_point
                    priority_ship.next_action = action if action != 'WAIT' else None
                    priority_ship.log.set_action, priority_ship.log.set_point = action, point

        # Ship building
        h2ns = [(p.halite, len(p.ships)) for p in self.board.players.values() if p.id is not me.id]
        nships_other = sorted(h2ns, key=lambda x: -x[0])[0][1]
        # TODO - +5 ships here is arbitrary
        should_still_spawn = ((len(me.ships) <= nships_other + 5) or (obs.step < 20)) \
                             and (obs.step < 360)
        reserve = config.convertCost if obs.step > 20 else 0
        for shipyard in me.shipyards:
            # If we can afford spawn, considering cumulation of other SY spawns and keeping a reserve for one yard.
            have_enough_halite = (me.halite - spawncount * config.spawnCost - reserve) >= config.spawnCost
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
    actions = myBot[obs.player].get_actions(obs, config)
    return actions
