from ..player import BasePlayer

class AlwaysDouble(BasePlayer):
    ''' 
    This player always selects a double piece if possible to make a move
    '''
    def __init__(self, name):
         super().__init__(f"AlwaysDouble::{name}")

    def filter(self, valids=None):
        valids = super().filter(valids)

        data = []
        for piece, head in valids:
            if piece[0] == piece[1]:
                data.append((piece, head))
        
        return data if data else valids
                
                    
