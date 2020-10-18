import os
from pathlib import Path

pathroot = Path(f'{os.environ["HOME"]}/work/kaggle/halite/')


def write_html(html, name):
    with open(pathroot / 'html' / name, 'w') as f:
        f.write(html)
