from ...domino import DominoManager
from ..mc import MCPlayer, MCSimulator
from ..player_view import PlayerView
from utils import game_data_collector, game_hand_builder, remaining_pieces
from math import log2


class _NNWrapper:
    def __init__(self, NN):
        self.NN = NN
        self.called = False
        self.v = None

    def __call__(self, state):
        a, b, v = self.NN(state)
        self.v = v
        self.called = True
        return a, b, v


class Simulator(MCSimulator):
    def __init__(self, name, mc_data, NN):
        super().__init__(f'AlphaZero::{name}')
        self.mc_data = mc_data
        self.NN = NN
        self.valids_per_state = []

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
        return  (Q + P) / (1 + N)

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

    def _filter(self):
        state, heads = self.get_utility()

        valids = []
        valids_mask = [0, 0]
        utility = (-float("inf"), 0)
        for i, piece in enumerate(self.pieces):
            bit = self.piece_bit(*piece)

            for h, (N, P, Q) in enumerate(heads):
                if not self.valid(piece, h):
                    continue
                valids_mask[h] += bit
                move_utility = (self.utility_value((N[i], P[bit], Q[i])), -N[i])
                if utility < move_utility:
                    utility = move_utility
                    valids = []
                if utility == move_utility:
                    valids.append((piece, h))
        self.states.append(state)
        self.valids_per_state.append(valids_mask)

        # //TODO: add randomness to ensure exploration

        return valids, valids_mask

    def filter(self, valids):
        valids, _ = self._filter()
        return valids

    def get_state(self, move):
        bit = int(log2(self.piece_bit(move)))
        state = self.states.pop(0)
        pos = bin(state)[::-1][:bit].count('1')
        return state, pos


class AlphaZero(MCPlayer):
    def __init__(self, name, handouts, rollouts, NN):
        super().__init__(f'AlphaZero::{name}', handouts, rollouts)
        self.NN = _NNWrapper(NN)
        self.states = []

    def filter(self, valids):
        # basic game information
        pieces, missing = game_data_collector(self.pieces, self.me, self.history)
        remaining = remaining_pieces(pieces, self.max_number)

        # State
        mc_state = {}

        # simultations
        for _ in range(self.handouts):
            fixed_hands = game_hand_builder(pieces, missing, remaining, self.pieces_per_player)
            hand = lambda: [PlayerView(h) for h in fixed_hands]
            for _ in range(self.rollouts):
                # rollout game configuration
                manager = DominoManager()
                players = [Simulator(name, mc_state, self.NN) for name in "0123"]
                manager.init(players, hand, self.max_number, self.pieces_per_player)

                # advance the game to the current state
                for e, *data in self.history:
                    if e.name == "MOVE":
                        move, _, head = data
                        manager.step(True, (move, head))
                    if e.name == "PASS":
                        manager.step(True, None)

                # run the rollout
                v = 0
                while True:
                    if manager.step():
                        w = manager.domino.winner
                        if w != -1:
                            v = [-1, 1][w == self.team]
                        break
                    if self.NN.called:
                        v = self.NN.v
                        break

                # update state
                for e, *data in manager.domino.logs[len(self.history):]:
                    if e.name == "MOVE":
                        move, id, head = data
                        state, pos = players[id].get_state(move)
                        N, _, Q = mc_state[state][head]
                        Q[pos] += v
                        N[pos] += 1
                        
        # store the current state & make a choice
        temp = Simulator("temp", mc_state, None)
        temp.reset(0, self.pieces, self.max_number)
        current_state = temp.encode_game()
        valids, valids_mask = temp._filter()
        self.states.append((current_state, *valids_mask))

        return valids
