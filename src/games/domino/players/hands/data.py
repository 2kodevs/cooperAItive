from ..player_view import PlayerView
from random import sample, choice

def data(max_number, pieces_per_player, high=True, number=-1):
    """
    Force player0 to have at least half of his pieces of the same number.
    Randomly distribute pieces among the other players.
    Valid pieces are all integer tuples of the form:
        (i, j) 0 <= i <= j <= max_number
    Each player will have `pieces_per_player`.
    """
    
    mid = (max_number + 1) // 2 + 1
    data_number = choice(list(range([0, mid][high], [mid, max_number + 1][high])))
    data_number = [data_number, number][number >= 0 and number <= max_number]
    cant = min(max_number + 1, pieces_per_player) // 2
    hand = [(min(data_number, i), max(data_number, i)) for i in sample(list(range(max_number + 1)), cant)]

    pieces = [(i, j) for i in range(max_number + 1) for j in range(max_number + 1) if i <= j and (i, j) not in hand]
    assert 4 * pieces_per_player <= len(pieces) + len(hand)
    
    hand.extend(sample(pieces, 4 * pieces_per_player - len(hand)))
    hands = [hand[i:i+pieces_per_player] for i in range(0, 4 * pieces_per_player, pieces_per_player)]
    return [PlayerView(h) for h in hands]

def data_low(max_number, pieces_per_player):
    return data(max_number, pieces_per_player, False)
