from ..player import BasePlayer
from ..utils import count_min

class Agachao(BasePlayer):
    ''' 
    This player doesn't want to pass, 
    so if there is a number that appears in only one piece, 
    this piece will not be used until there's no other choice.
    '''
    def __init__(self, name):
         super().__init__(f"Agachao::{name}")

    def filter(self, valids=None):
        valids = super().filter(valids)

        heads = []
        first_move = True
        amount_played = {}

        for e, *d in self.history:
            if e.name == 'MOVE':
                player, piece, head = d
                if player == self.me:
                    amount_played[piece[0]] = amount_played.get(piece[0], 0) + 1
                    if piece[0] != piece[1]:
                        amount_played[piece[1]] = amount_played.get(piece[1], 0) + 1
                if first_move:
                    heads = list(piece)
                    first_move = False
                else:
                    heads[head] = piece[piece[0] == heads[head]]

        best, data = (-1, -1), []
        for piece, head in valids:
            mn, i = count_min(self, piece)
            value = (mn, amount_played.get(piece[i], 0))
            if value > best:
                best = value
                data.clear()
            if value == best:
                data.append((piece, head))

        return data
                
                    
