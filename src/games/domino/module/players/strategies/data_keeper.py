from ..player import BasePlayer

class DataKeeper(BasePlayer):
    '''
    Player who keeps his most repeated pieces.
    '''
    def __init__(self, name):
        super().__init__(f"DataKeeper::{name}")

    def filter(self, valids=None):
        valids = super().filter(valids)

        datas = {}
        for p1, p2 in self.pieces:
            if p1 != p2:
                datas[p2] = datas.get(p2, 0) + 1
            datas[p1] = datas.get(p1, 0) + 1
                
        best, selected = float('inf'), []
        for piece, head in valids:
            value = max(datas[piece[0]], datas[piece[1]])

            if value < best:
                best = value
                selected.clear()
            if value == best:
                selected.append((piece, head))

        return selected
        