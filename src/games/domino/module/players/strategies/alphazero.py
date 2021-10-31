from ..player import BasePlayer
from .utils.alphazero import encoder_generator, rollout_maker, selector_maker
from .utils.mc import monte_carlo
from .models import AlphaZeroNet
from .utils import parse_bool


class AlphaZero(BasePlayer):
    def __init__(self, name, handouts, rollouts, NN, coop = 1, cput = 1):
        super().__init__(f'AlphaZero::{name}')

        if isinstance(NN, str):
            self.NN = AlphaZeroNet.load(NN)
        else: 
            self.NN = NN
        self.handouts = int(handouts)
        self.rollouts = int(rollouts)
        self.coop = coop
        self.cput = cput

    def filter(self, valids):
        data = {}
        selector = selector_maker(data, self.valid_moves(), self.pieces_per_player - len(self.pieces), False, 6)
        encoder = encoder_generator(self.max_number)
        rollout = rollout_maker(data, self.NN, self.coop, self.cput)

        _, action, *_ = monte_carlo(
            self, 
            encoder, 
            rollout, 
            selector,
            self.handouts,
            self.rollouts,
        )

        return [action]
