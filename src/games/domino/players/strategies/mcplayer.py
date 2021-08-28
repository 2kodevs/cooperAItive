from ..player import BasePlayer
from utils import game_data_collector, game_hand_builder, remaining_pieces


class MonteCarlo(BasePlayer):
    def __init__(self, name, handouts=10, rollouts=10):
        super().__init__(f'MonteCarlo::{name}')
        self.set_simulations(handouts, rollouts)

    def set_simulations(self, handouts, rollouts):
        self.handouts = handouts
        self.rollouts = rollouts

    def filter(self, valids):
        # basic game information
        pieces, missing = game_data_collector(self.pieces, self.me, self.history)
        remaining = remaining_pieces(pieces, 10) # //TODO: Modify BasePlayer to save the max number

        # State & Neural Network
        state = {}
        NN = None

        # simultations
        for _ in range(self.handouts):
            hands = game_hand_builder(pieces, missing, remaining, self.pieces_per_player)
            for _ in range(self.rollouts):
                pass       
