from .table_counter import TableCounter
from ..player import BasePlayer

class LessPlayed(TableCounter):
    ''' Always select the piece less played
    '''
    def __init__(self, name):
        super().__init__(name)
        self.name = f'LessPlayed::{name}'

    def filter(self, valids=None):
        valids = BasePlayer.filter(self, valids)

        cant = self.count_table()
        best, data = float('inf'), []
        for piece, head in valids:
            value = cant.get(piece[piece[0] == self.heads[head]], 0)
            if value < best:
                best, data = value, []
            if value == best:
                data.append((piece, head))
        return valids
