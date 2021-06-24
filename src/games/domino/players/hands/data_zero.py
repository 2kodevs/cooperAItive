from ..player_view import PlayerView
from random import sample

def data_zero(max_number, pieces_per_player):
    """
    Force player0 to have at least half of his pieces of zero number, including double zero.
    Randomly distribute pieces among the other players.
    Valid pieces are all integer tuples of the form:
        (i, j) 0 <= i <= j <= max_number
    Each player will have `pieces_per_player`.
    """
    
    data_number = 0
    cant = (min(max_number + 1, pieces_per_player) // 2)
    hand = [(data_number, i) for i in sample(list(range(max_number + 1)), cant)]

    if not (0, 0) in hand:
        hand.pop()
        hand.append((0, 0))
    
    pieces = [(i, j) for i in range(max_number + 1) for j in range(max_number + 1) if i <= j and (i, j) not in hand]
    assert 4 * pieces_per_player <= len(pieces) + len(hand)
    
    hand.extend(sample(pieces, 4 * pieces_per_player - len(hand)))
    hands = [hand[i:i+pieces_per_player] for i in range(0, 4 * pieces_per_player, pieces_per_player)]
    return [PlayerView(h) for h in hands]
