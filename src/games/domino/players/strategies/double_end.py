from ..player import BasePlayer
from .utils import is_end

class DoubleEnd(BasePlayer):
    ''' 
    This player always selects a double piece if possible to make a move at the end of the game.
    '''
    def __init__(self, name):
         super().__init__(f"DoubleEnd::{name}")

    def filter(self, valids=None):
        valids = super().filter(valids)

        data = []
        for piece, head in valids:
            if piece[0] == piece[1]:
                data.append((piece, head))

        if data and is_end(self.history, self.pieces_per_player):
            return data
        return valids
                
                    
