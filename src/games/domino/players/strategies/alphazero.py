

from ..player import BasePlayer
from .utils.alphazero import encoder_generator, rollout_maker
from .utils.mc import monte_carlo


class AlphaZero(BasePlayer):
    def __init__(self, name, handouts, rollouts, NN):
        super().__init__(f'AlphaZero::{name}', handouts, rollouts)
        self.NN = NN
        self.handouts = handouts
        self.rollouts = rollouts

    def filter(self, valids):
        selector = None # //TODO: Create a real selector
        encoder = encoder_generator(self.max_number)
        rollout = rollout_maker({}, self.NN)

        _, action, _ = monte_carlo(
            self, 
            encoder, 
            rollout, 
            selector,
            self.handouts,
            self.rollouts,
        )

        return [action]
