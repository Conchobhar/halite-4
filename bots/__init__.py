from pathlib import Path
import importlib
import importlib.util


def latest():
    modules = Path(__file__).glob('v*py')
    module_path = max(modules).as_posix()
    spec = importlib.util.spec_from_file_location("bot", module_path)
    bot = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(bot)
    return bot.agent
