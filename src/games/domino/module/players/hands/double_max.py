from ..player_view import PlayerView
from random import sample

def double_max(max_number, pieces_per_player):
    """
    Force player0 to have the larger double.
    Randomly distribute pieces among the other players.
    Valid pieces are all integer tuples of the form:
        (i, j) 0 <= i <= j <= max_number
    Each player will have `pieces_per_player`.
    """
    hand = [(max_number, max_number)]
    
    pieces = [(i, j) for i in range(max_number + 1) for j in range(max_number + 1) if i <= j and (i, j) not in hand]
    assert 4 * pieces_per_player <= len(pieces) + 1
    
    hand.extend(sample(pieces, 4 * pieces_per_player - 1))
    hands = [hand[i:i+pieces_per_player] for i in range(0, 4 * pieces_per_player, pieces_per_player)]
    return [PlayerView(h) for h in hands]
