from ..player_view import PlayerView
from .utils import share_data

def data_partner(max_number, pieces_per_player, high=True):
    """
    Share a high data or a low data with partner
    Randomly distribute pieces among others player.
    Valid pieces are all integer tuples of the form:
        (i, j) 0 <= i <= j <= max_number
    Each player will have `pieces_per_player`.
    """
    hand0, hand2, temp_hands = share_data(max_number, pieces_per_player, high)

    hands = [hand0, temp_hands[0], hand2, temp_hands[1]]

    return [PlayerView(h) for h in hands]

def data_partner_low(max_number, pieces_per_player):
    return data_partner(max_number, pieces_per_player, high=False)
