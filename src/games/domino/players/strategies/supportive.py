from ..player import BasePlayer

class Supportive(BasePlayer):
    '''
    When the other player of the team is the hand, plays for him.
    '''
    def __init__(self, name):
        super().__init__(f"Supportive::{name}")

    def filter(self, valids=None):
        valids = super().filter(valids)
        
        heads = []
        passed = {}
        first_move = True
        player_pieces = {}
        my_pieces, partner_pieces = 0, 0
        for e, *d in self.history:
            if e.name == 'MOVE':
                player, piece, head = d
                if first_move:
                    heads = list(piece)
                    first_move = False
                else:
                    heads[head] = piece[piece[0] == heads[head]]
                    my_pieces += (player == self.me)
                    partner_pieces += (player == self.partner)
                    if not player_pieces.get(heads[head]):
                        player_pieces[heads[head]] = player
            elif e.name =='PASS' and d[0] == self.partner:
                h0, h1 = heads
                passed[h0] = True
                passed[h1] = True

        #True if current_player is the hand
        if partner_pieces <= my_pieces:
            return valids

        top = []
        medium = []
        low = []
        for piece, head in valids:
            next_head = piece[piece[0] == self.heads[head]]
            if passed.get(self.heads[head]):
                top.append((piece, head))
            elif player_pieces.get(self.heads[head]) == self.partner:
                low.append((piece, head))
            elif player_pieces.get(next_head) == self.partner:
                top.append((piece, head))
            else:
                medium.append((piece, head))

        for data in [top, medium, low]:
            if data: 
                return data
