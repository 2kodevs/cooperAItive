from .player_view import PlayerView
from .player import BasePlayer
from .random import Random
from .monte_carlo import MonteCarlo
from .alphazero import AlphaZero
from .human import Human
from .hands import *
from .models import AlphaZeroNet
from .utils import alphazero as alphazero_utils, mc as mc_utils, game as game_utils


PLAYERS = [
    Random,
    MonteCarlo,
    AlphaZero,
    Human,
]


def get_player(value, elements=PLAYERS):
    value = value.lower()
    for obj in elements:
        if obj.__name__.lower() == value:
            return obj
        
    raise ValueError(f"{value} not found in {[e.__name__ for e in elements]}")
    