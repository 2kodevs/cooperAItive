from ..mc import MCPlayer
from utils import game_data_collector, game_hand_builder, remaining_pieces


class MonteCarlo(MCPlayer):
    def __init__(self, name, handouts=10, rollouts=10):
        super().__init__(f'Cooperative.v1::{name}', handouts, rollouts)

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
