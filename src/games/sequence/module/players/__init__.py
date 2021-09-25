from .player_view import PlayerView
from .player import BasePlayer
from .random import Random


PLAYERS = [
    Random,
]


def get_player(value, elements=PLAYERS):
    value = value.lower()
    for obj in elements:
        if obj.__name__.lower() == value:
            return obj
        
    raise ValueError(f"{value} not found in {[e.__name__ for e in elements]}")
    