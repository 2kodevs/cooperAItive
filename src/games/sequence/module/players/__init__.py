from .utils import alphazero as alphazero_utils, mc as mc_utils, game as game_utils
from .player_view import PlayerView
from .monte_carlo import MonteCarlo
from .alphazero import AlphaZero
from .models import AlphaZeroNet
from .heuristic import Heuristic
from .player import BasePlayer
from .random import Random
from .human import Human
from .hands import *


class Shortcut:
    def __init__(self, name, cls):
        self.__name__ = name
        self.cls = cls

    def __call__(self, *args, **kwds):
        return self.cls(*args, **kwds)


PLAYERS = [
    Random,
    MonteCarlo,
    AlphaZero,
    Human,
    Heuristic,
    Shortcut("MC", MonteCarlo),
    Shortcut("A0", AlphaZero),
]


def get_player(value, elements=PLAYERS):
    value = value.lower()
    for obj in elements:
        if obj.__name__.lower() == value:
            return obj
        
    raise ValueError(f"{value} not found in {[e.__name__ for e in elements]}")
