from ..mc import MCPlayer, MCSimulator
from utils import game_data_collector, game_hand_builder, remaining_pieces


class Simulator(MCSimulator):
    def __init__(self, name, mc_data, NN):
        super().__init__(f'AlphaZero::{name}')
        self.mc_data = mc_data
        self.NN = NN

    def piece_bit(self, a, b):
        return 1 << (a * self.max_number + b)

    def encode_game(self):
        pieces_mask = 0
        for p in self.pieces:
            pieces_mask += self.piece_bit(*p)
        player = (pieces_mask, 1 << self.me, 0)

        history = []
        for e, *data in self.history:
            if e.name == 'MOVE':
                move, id, head = data
                history.append((self.piece_bit(*move), 1 << id, head))
            if e.name == 'PASS':
                history.append((0, 0, 0))
            
        return [pieces_mask, *history]

    def utility_value(self, data):
        N, P, Q = data
        return Q + (P / (1 + N))

    def get_utility(self):
        state = self.encode_game()
        sz = len(self.pieces)
        if state not in self.mc_data:
            p0, p1, _ = self.NN(state)
            self.mc_data[state] = [
                ([0] * sz, p0, [0] * sz),
                ([0] * sz, p1, [0] * sz),
            ]
        return state, self.mc_data[state]

    def filter(self, valids):
        state, heads = self.get_utility()

        self.states.append(state)
        valids = []
        utility = (-float("inf"), 0)
        for i, piece in enumerate(self.pieces):
            bit = self.piece_bit(*piece)

            for h, (N, P, Q) in enumerate(heads):
                if not self.valid(piece, h):
                    continue
                move_utility = (self.utility_value((N[i], P[bit], Q[i])), -N[i])
                if utility < move_utility:
                    utility = move_utility
                    valids = []
                if utility == move_utility:
                    valids.append((piece, h))

        # //TODO: add randomness to ensure exploration

        return valids


class AlphaZero(MCPlayer):
    def __init__(self, name, handouts=10, rollouts=10):
        super().__init__(f'AlphaZero::{name}', handouts, rollouts)

    def filter(self, valids):
        # basic game information
        pieces, missing = game_data_collector(self.pieces, self.me, self.history)
        remaining = remaining_pieces(pieces, self.max_number)

        # State & Neural Network
        state = {}
        NN = None

        # simultations
        for _ in range(self.handouts):
            hands = game_hand_builder(pieces, missing, remaining, self.pieces_per_player)
            for _ in range(self.rollouts):
                pass       
