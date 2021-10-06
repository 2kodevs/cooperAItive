from .hand_out import handout
from .utils import *


HANDS = [
    handout,
]


def get_hand(value, elements=HANDS):
    value = value.lower()
    for obj in elements:
        if obj.__name__.lower() == value:
            return obj
        
    raise ValueError(f"{value} not found in {[e.__name__ for e in elements]}")