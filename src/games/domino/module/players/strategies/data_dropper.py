from ..player import BasePlayer

class DataDropper(BasePlayer):
    '''
    Player who drops his most repeated pieces.
    '''
    def __init__(self, name):
        super().__init__(f"DataDropper::{name}")

    def filter(self, valids=None):
        valids = super().filter(valids)

        datas = {}
        for p1, p2 in self.pieces:
            if p1 != p2:
                datas[p2] = datas.get(p2, 0) + 1
            datas[p1] = datas.get(p1, 0) + 1

        pieces = {x for p, _ in valids for x in p}
        bigger_data = max([c for v, c in datas.items() if v in pieces])
                
        best, selected = ([-1], -1), []
        for piece, head in valids:
            heads = self.heads[:]
            if heads == [-1, -1]: heads = list(piece)
            else: heads[head] = piece[piece[0] == heads[head]]
            values = [datas.get(num, 0) for num in heads]
            values.sort(reverse=True)
            value = (values, datas.get(self.heads[head], 0) != bigger_data)
            
            if value > best:
                best = value
                selected.clear()
            if value == best:
                selected.append((piece, head))

        return selected
        
