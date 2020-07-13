from kaggle_environments.envs.halite.helpers import Board, ShipAction, ShipyardAction


def my_agent(obs, config):
    board = Board(obs, config)
    me = board.current_player
    while len(me.next_actions) != len(me.ships):
        # Ship building
        for shipyard in me.shipyards:
            ...
        for ship in me.ships:
            # If first turn, create dock
            # TODO - if ndocks < 1, prioritize building dock
            if obs.step == 0:
                ship.next_action = ShipAction.CONVERT
                return me.next_actions
            else:
                ship.next_action = ShipAction.NORTH
    return me.next_actions




def render(self, **kwargs):
    """
    Renders a visual representation of the current state of the environment.

    Args:
        mode (str): html, ipython, ansi, human (default)
        **kwargs (dict): Other args are directly passed into the html player.

    Returns:
        str: html if mode=html or ansi if mode=ansi.
        None: prints ansi if mode=human or prints html if mode=ipython
    """
    mode = get(kwargs, str, "human", path=["mode"])
    if mode == "ansi" or mode == "human":
        args = [self.state, self]
        out = self.renderer(*args[:self.renderer.__code__.co_argcount])
        if mode == "ansi":
            return out
        print(out)
    elif mode == "html" or mode == "ipython":
        window_kaggle = {
            "debug": get(kwargs, bool, self.debug, path=["debug"]),
            "autoplay": get(kwargs, bool, self.done, path=["autoplay"]),
            "step": 0 if get(kwargs, bool, self.done, path=["autoplay"]) else (len(self.steps) - 1),
            "controls": get(kwargs, bool, self.done, path=["controls"]),
            "environment": self.toJSON(),
            **kwargs,
        }
        player_html = get_player(window_kaggle, self.html_renderer)
        if mode == "html":
            return player_html
        from IPython.display import display, HTML
        player_html = player_html.replace('"', '&quot;')
        width = get(kwargs, int, 300, path=["width"])
        height = get(kwargs, int, 300, path=["height"])
        html = f'<iframe srcdoc="{player_html}" width="{width}" height="{height}" frameborder="0"></iframe> '
        if "return_obj" in kwargs: return html
        display(HTML(html))
    elif mode == "json":
        return json.dumps(self.toJSON(), sort_keys=True)
    else:
        raise InvalidArgument(
            "Available render modes: human, ansi, html, ipython")