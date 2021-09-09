from ..player import BasePlayer

class Casino(BasePlayer):
    ''' Play doble white if possible as the first move of the game
    '''
    def __init__(self, name):
        super().__init__(f'Casino::{name}')

    def filter(self, valids=None):
        valids = super().filter(valids)

        if self.heads != [-1, -1]:
            return valids
        if (0, 0) in self.pieces:
            return [((0, 0), 0)]
        return valids
    