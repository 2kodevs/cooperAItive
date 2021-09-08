from ..player_view import PlayerView
from random import sample

def no_doubles(max_number, pieces_per_player):
    """
    Player 0 will have no doubles.
    Randomly distribute pieces among every player.
    Valid pieces are all integer tuples of the form:
        (i, j) 0 <= i <= j <= max_number
    Each player will have `pieces_per_player`.
    """
    all_pieces = {(i, j) for i in range(max_number + 1) for j in range(max_number + 1) if i <= j}
    pieces0 = all_pieces - {(i, i) for i in range(max_number + 1)}
    hand0 = sample(pieces0, pieces_per_player)

    pieces = list(all_pieces - set(hand0))

    assert 4 * pieces_per_player <= len(pieces) + len(hand0)
    hand = sample(pieces, 3 * pieces_per_player)
    hands = [hand0]
    hands += [hand[i:i+pieces_per_player] for i in range(0, 3 * pieces_per_player, pieces_per_player)]

    return [PlayerView(h) for h in hands]