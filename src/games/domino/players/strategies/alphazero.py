

from ..player import BasePlayer
from .utils.alphazero import encoder_generator, rollout_maker, selector_maker
from .utils.mc import monte_carlo


class AlphaZero(BasePlayer):
    def __init__(self, name, handouts, rollouts, NN):
        super().__init__(f'AlphaZero::{name}', handouts, rollouts)
        self.NN = NN
        self.handouts = handouts
        self.rollouts = rollouts

    def filter(self, valids):
        data = {}
        selector = selector_maker(data, self.valid_moves(), self.pieces_per_player - len(self.pieces), False)
        encoder = encoder_generator(self.max_number)
        rollout = rollout_maker(data, self.NN)

        _, action, _ = monte_carlo(
            self, 
            encoder, 
            rollout, 
            selector,
            self.handouts,
            self.rollouts,
        )

        return [action]
