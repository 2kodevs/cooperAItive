from ..player import BasePlayer

class Passer(BasePlayer):
    '''
   Always tries the next player passes
    '''
    def __init__(self, name):
        super().__init__(f"Passer::{name}")

    def filter(self, valids=None):
        valids = super().filter(valids)
        
        heads = []
        next_player_passed = {}
        first_move = True
        for e, *d in self.history:
            if e.name == 'MOVE':
                player, piece, head = d
                if first_move:
                    heads = list(piece)
                    first_move = False
                else:
                    heads[head] = piece[piece[0] == heads[head]]
            elif e.name =='PASS' and d[0] == self.next:
                h0, h1 = heads
                next_player_passed[h0] = True
                next_player_passed[h1] = True

        best, selected = -2, []
        for piece, head in valids:
            next_head = piece[piece[0] == self.heads[head]]
            value = 0
            value -= next_player_passed.get(self.heads[head], 0)
            value += 2 * next_player_passed.get(next_head, 0)
            if value > best:
                best = value
                selected.clear()
            if value == best:
                selected.append((piece, head))

        return selected
