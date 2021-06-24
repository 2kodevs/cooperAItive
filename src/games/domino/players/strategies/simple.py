from ..player import BasePlayer

class BigDrop(BasePlayer):
    """ Always drop piece with highest score
    """
    def __init__(self, name):
        super().__init__(f"BigDrop::{name}")

    def filter(self, valids=None):
        valids = super().filter(valids)

        max_weight = 0
        fat = []

        for piece, head in valids:
            weight = piece[0] + piece[1]

            if weight > max_weight:
                fat.clear()
                max_weight = weight

            if weight == max_weight:
                fat.append((piece, head))

        return fat


class Random(BasePlayer):
    """ Make a random move at each step
    """
    def __init__(self, name):
        super().__init__(f"Random::{name}")


class Frequent(BasePlayer):
    """ Find piece most frequent in its hand. It tries to avoid passing.
    """
    def __init__(self, name):
        super().__init__(f"Frequent::{name}")

    def filter(self, valids=None):
        valids = super().filter(valids)
        # One piece A is neighbor of B if have at least one common number
        # Find pieces with largest number of neighbors
        pieces = []
        best_freq = -1

        for (cur_piece, head) in valids:
            freq = 0

            for piece in self.pieces:
                if piece[0] in cur_piece or piece[1] in cur_piece:
                    freq += 1

            if freq > best_freq:
                best_freq = freq
                pieces = []

            if freq == best_freq:
                pieces.append((cur_piece, head))

        # Return one piece with largest number of neighbors randomly
        return pieces
