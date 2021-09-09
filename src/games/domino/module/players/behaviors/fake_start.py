from ..player import BasePlayer
from ..utils import count_min

class FakeStart(BasePlayer):
    ''' Select the pieces that have a number that is unique in the hand
    '''
    def __init__(self, name):
        super().__init__(f"FakeStart::{name}")

    def filter(self, valids=None):
        valids = super().filter(valids)

        moves = sum(e.name in ['MOVE', 'PASS'] for e, *_ in self.history)
        if moves < 4:
            data = []
            for piece, head in valids:
                val, _ = count_min(self, piece)
                if val == 1:
                    data.append((piece, head))
            if data: return data
        return valids
        
