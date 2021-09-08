from ..player import BasePlayer

class Repeater(BasePlayer):
    """ Find the piece with the number more used by himself. It tries to avoid passing.
    """
    def __init__(self, name):
        super().__init__(f"Repeater::{name}")


    def times_played(self):
        ''' Given a list of numbers return the amount of repetions per each one
        '''

        def update(d, player, heads):
            if player == self.position:
                for num in heads:
                    d[num] = d.get(num, 0) + 1
        
        times = {}
        all_moves = [d for e, *d in self.history if e.name == 'MOVE']
        if all_moves:
            first, *moves = all_moves
            heads = list(first[1])
            update(times, first[0], heads)
            for data in moves:
                player, piece, head = data
                heads[head] = piece[piece[0] == heads[head]]
                update(times, player, heads)    
        return times


    def filter(self, valids=None):
        valids = super().filter(valids)
        # Select the movement that generate the maximun repetion
        best, selected = [-1, -1], []
        times = self.times_played()
        for piece, head in valids:
            heads = self.heads[:]
            heads[head] = piece[piece[0] == heads[head]]
            heads_value = [times.get(num, 0) for num in heads]
            heads_value.sort(reverse=True)
            if heads_value > best:
                best, selected = heads_value, []
            if heads_value == best:
                selected.append((piece, head))

        return selected
