from ..player import BasePlayer

class SmallDrop(BasePlayer):
    """ Always drop piece with lowest score
    """
    def __init__(self, name):
        super().__init__(f"SmallDrop::{name}")

    def filter(self, valids=None):
        valids = super().filter(valids)

        min_weight = float('inf')
        fat = []

        for piece, head in valids:
            weight = piece[0] + piece[1]

            if weight < min_weight:
                fat.clear()
                min_weight = weight

            if weight == min_weight:
                fat.append((piece, head))

        return fat
