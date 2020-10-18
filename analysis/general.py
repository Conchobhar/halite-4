"""Scratch pad"""
import json
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.utils import structify
from kaggle_environments.envs.halite.helpers import *
from utils.base import write_html
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# from agent.base import agent  # CONFIG - MANUALLY IMPORT BOT
myid = 1
# CONFIG - REPLAY AND CORRESPONDING ID
pathroot = Path(f'{os.environ["HOME"]}work/kaggle/halite/')
pathdl = Path(f'{os.environ["HOME"]}Downloads/')
# path_replay = pathdl / '1984103.json'
paths = pathdl.glob('*.json')
path = max(paths, key=lambda p:p.stat().st_ctime)


def load_latest():
    paths = pathdl.glob('*.json')
    path = max(paths, key=lambda p: p.stat().st_ctime)
    with open(path, 'r') as f:
        match = json.load(f)
    env = make("halite", configuration=match['configuration'], steps=match['steps'])
    config = env.configuration
    t = env.render(mode='html', return_obj=True, width=800, height=800, header=False, controls=True)
    write_html(t, 'replay.html')
    return match, config


def load_latest_ship_attr(mainattr='ships', attr='halite'):
    match, config = load_latest()
    record = {}
    for pid in range(4):
        record[pid] = {}
    for step in range(400):
        state = match['steps'][step][0]  # list of length 1 for each step
        obs = state['observation']  # these are observations at this step
        for pid in range(4):
            qps = get_quadrant_points(pid)
            obs['player'] = pid  # change the player to the one we want to inspect
            precord = record[pid]
            obs = structify(obs)  # turn the dict's into structures with attributes
            b = Board(obs, config)
            iterable = getattr(b, mainattr).values() if mainattr == 'cells' else b.me
            keyattr = 'position' if mainattr == 'cells' else 'id'
            for s in iterable:
                if attr == 'position' and s.halite > 0:
                    continue
                if mainattr == 'cells' and s.halite == 0 or s.position not in qps:
                    continue
                key = getattr(s, keyattr)
                if key not in precord:
                    precord[key] = {}
                    precord[key][step] = getattr(s, attr)
                else:
                    precord[key][step] = getattr(s, attr)
    return record


pid = 0
pid2quadrant = {
    0: Point(5, 15),
    1: Point(15, 15),
    2: Point(5, 5),
    3: Point(15, 5)
}


def dist(p1, p2, dim=21):
    p = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
    x, y = min(p[0], dim - p[0]), min(p[1], dim - p[1])
    return abs(sum([x, y]))

def get_quadrant_points(pid):
    """Define player quadrant incl. shared points."""
    points = []
    qp = (10 // 2) + 1
    quadrant_position = pid2quadrant[pid]
    for x in range(-qp, qp + 1):
        for y in range(-qp, qp + 1):
            points.append(quadrant_position.translate(Point(x, y), 21))
    return points




# Analysis
pid2col = {
    0: 'y',
    1: 'red',
    2: 'green',
    3: 'purple',
}

shalite = load_latest_ship_attr('ships', 'halite')
cells = load_latest_ship_attr('cells', 'halite')

spos2 = load_latest_ship_attr('position')

print('data loaded.')


fig, ax = plt.subplots(figsize=(8,8))
for pid in range(4):
    df = pd.DataFrame.from_dict(shalite[pid])
    zero = df.applymap(lambda x: x == 0).sum(axis=1)
    gtzero = df.applymap(lambda x: x > 0).sum(axis=1)
    zero = zero.rolling(window=20).mean()
    gtzero = gtzero.rolling(window=20).mean()
    (gtzero/(zero+gtzero)).plot(ax=ax, color=pid2col[pid])
ax.legend([x for x in range(4)])

df = pd.DataFrame(cells[2])
df[0:10].head()
df[0:400].plot()

# For cells - min halite per turn over quadrant
df.apply(np.mean, axis=1).plot()
df.apply(np.std, axis=1).plot()
(df.apply(np.mean, axis=1) - df.apply(np.std, axis=1)).plot()


def get_min_harv_amount_by_step():
    step = 0
    if step > 250:
        return 40
    else:  # Return a quadratic of yours
        a = 0.0025
        c = 40
        return a * step ** + step + c



pleaderboard = pathroot / 'analysis/publicleaderboarddata.zip'
dflb = pd.read_csv(pleaderboard)

lbnames_top5 = ('convexOptimization' ,'Tom Van de Wiele', 'Leukocyte', 'mzotkiew', 'Robiland', 'Fei Wang')


# Top 5 score plot
dflb[dflb.TeamName.isin(lbnames_top5)].pivot_table(values='Score', columns='TeamName', index='SubmissionDate').plot()

df.applymap(lambda x:(x.x,x.y) if x is not np.nan else x)['2-1'].plot()
df.loc[0:400].plot(marker='o')

cdiff = df.loc[0:400].diff()
cdiff.applymap(lambda x:float('nan') if x <= 0 else x).plot()
_f = cdiff.apply(lambda x:x > 0)
cdiff[_f].fillna(-10).plot(kind='hist')


ratio = (cdiff/df)
ratio = ratio[ratio.applymap(lambda x:x < 0.1)]
ratio.plot()

"""if tcell.halite > 40"""





df = pd.DataFrame.from_dict(spos[0])
x = df.applymap(lambda p: p.x if isinstance(p, Point) else p)
y = df.applymap(lambda p: p.y if isinstance(p, Point) else p)

fig, ax = plt.subplots()
cols = x.columns[1:3]

# line, _ = ax.plot(x[cols][0:30], y[cols][0:30], marker='o')




cols = x.columns
fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim([0,20])
ax.set_ylim([0,20])
ax.set_xticks([x-0.5 for x in range(0,20)])
ax.set_yticks([x-0.5 for x in range(0,20)])
plt.grid(b=True, which='both', color='#aaaaaa', linestyle='-')
lines = ax.plot(x[cols][0:1], y[cols][0:1], marker='o')


def init():  # only required for blitting to give a clean slate.
    for n, line in enumerate(lines):
        line.set_ydata([np.nan] * len(x))
        line.set_xdata([np.nan] * len(x))
    return lines


def animate(i,span=10):
    for n, line in enumerate(lines):
        line.set_xdata(x[cols[n]][i:i+span])
        line.set_ydata(y[cols[n]][i:i+span])
    return lines

ani = animation.FuncAnimation(
    fig, animate, init_func=init, frames=400, interval=16, blit=True, save_count=50, repeat=False)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()


hlt = np.zeros((21,21))
for (x, y), h in self.harvest_spot_values:
    hlt[y, x] = round(h, 1)
hlt = np.flip(hlt,0)
fig, ax = plt.subplots()
im = ax.imshow(hlt)

# Loop over data dimensions and create text annotations.
for i in range(21):
    for j in range(21):
        text = ax.text(j, i, round(hlt[i, j]),
                       ha="center", va="center", color="w")
plt.axis('off')
# ax.set_xticks(range(21))
# ax.set_yticks([str(x) for x in range(21, 0, -1)])