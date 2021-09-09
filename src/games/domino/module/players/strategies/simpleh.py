from ..player import BasePlayer
import numpy as np


# TODO: Explain how this player was implemented and how was found the set of parameters.
class SimpleHybrid(BasePlayer):
    PARAMETERS = [0.613, 0.075, 0.171, 0.562, 0.007, 2.127]

    def __init__(self, name, coef=None):
        super().__init__(f"SmartH::{name}")

        if coef is None:
            coef = SimpleHybrid.PARAMETERS

        self.coef = coef

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name

    def eval_random(self, piece):
        return 1

    def eval_big_drop(self, piece):
        sums = []
        bigger = 0
        count = 1

        for _piece in self.pieces:
            if _piece[0] + _piece[1] > bigger:
                bigger = _piece[0] + _piece[1]
                count = 1
            if _piece[0] + _piece[1] == bigger:
                count += 1

        if piece[0] + piece[1] == bigger:
            return 1. / count
        else:
            return 0

    def eval_big_drop_soft(self, piece):
        return piece[0] + piece[1]

    def eval_frequent(self, piece):
        bigger = 0
        total = []

        for i in self.pieces:
            count = 0
            for j in self.pieces:
                if i[0] == j[0] or i[0] == j[1] or i[1] == j[0] or i[1] == j[1]:
                    count += 1
            total.append(count)

            if count > bigger:
                bigger = count

        _count = 0
        for i in total:
            if i == bigger:
                _count += 1

        index = self.pieces.index(piece)
        if total[index] == bigger:
            return 1 / _count
        else:
            return 0

    def eval_frequent_soft(self, piece):
        count = 0

        for _piece in self.pieces:
            if piece[0] == _piece[0] or piece[0] == _piece[1] or \
                    piece[1] == _piece[0] or piece[1] == _piece[1]:
                count += 1

        return count

    def eval_doubles(self, piece):
        if piece[0] == piece[1]:
            return 1
        else:
            return 0

    def filter(self, valids=None):
        valids = super().filter(valids)

        heads = self.heads
        bigger = float('-inf')
        final_piece = None

        final_pieces = []
        for piece, _ in valids:
            values = []

            values.append(self.eval_random(piece))
            values.append(self.eval_big_drop(piece))
            values.append(self.eval_big_drop_soft(piece))
            values.append(self.eval_frequent(piece))
            values.append(self.eval_frequent_soft(piece))
            values.append(self.eval_doubles(piece))

            mul = np.multiply(self.coef, values)
            val = np.sum(mul)

            if val > bigger:
                bigger = val
                final_pieces = []
            if val == bigger:
                final_pieces.append((piece, [1, 0][heads[0] in piece]))

        return final_pieces
