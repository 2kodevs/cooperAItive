from ..player import BasePlayer

class TableCounter(BasePlayer):
    ''' 
    Select the pice with higher score from the pices with most frequent played values
    '''
    def __init__(self, name):
         super().__init__(f"TableCounter::{name}")


    def count_table(self):
        cant = {}
        pieces = [d[1] for e, *d in self.history if e.name == 'MOVE']
        for p in pieces:
            cant[p[0]] = cant.get(p[0], 0) + 1
            if p[0] != p[1]:
                cant[p[1]] = cant.get(p[1], 0) + 1
        return cant

    
    def filter(self, valids=None):
        valids = super().filter(valids)

        best, data = -1, []
        cant = self.count_table()
        for piece, head in valids:
            value = cant.get(piece[piece[0] == self.heads[head]], 0)
            if value > best:
                best, data = value, []
            if value == best:
                data.append((piece, head))
               
        return data
