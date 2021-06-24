from ..player_view import PlayerView
from random import sample

def hand_out(max_number, pieces_per_player):
    """
    Randomly distribute pieces among every player.
    Valid pieces are all integer tuples of the form:
        (i, j) 0 <= i <= j <= max_number
    Each player will have `pieces_per_player`.
    """
    pieces = [(i, j) for i in range(max_number + 1) for j in range(max_number + 1) if i <= j]
    assert 4 * pieces_per_player <= len(pieces)
    hand = sample(pieces, 4 * pieces_per_player)
    hands = [hand[i:i+pieces_per_player] for i in range(0, 4 * pieces_per_player, pieces_per_player)]
    return [PlayerView(h) for h in hands]
