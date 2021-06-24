from ..player_view import PlayerView
from .utils import share_data
from random import shuffle

def data_opponent(max_number, pieces_per_player, high=True):
    """
    Share a high data or a low data with a oponent
    Randomly distribute pieces among other player.
    Valid pieces are all integer tuples of the form:
        (i, j) 0 <= i <= j <= max_number
    Each player will have `pieces_per_player`.
    """
    hand0, handO, temp_hands = share_data(max_number, pieces_per_player, high)

    oponent_hands = [handO, temp_hands[0]]
    shuffle(oponent_hands)

    hands = [hand0, oponent_hands[0], temp_hands[1], oponent_hands[1]]

    return [PlayerView(h) for h in hands]

def data_opponent_low(max_number, pieces_per_player):
    return data_opponent(max_number, pieces_per_player, high=False)
    
