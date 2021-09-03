from ..player import BasePlayer
from .utils.alphazero import encoder_generator
from .utils.mc import monte_carlo, rollout_maker, selector_generator


class MonteCarlo(BasePlayer):
    def __init__(self, name, handouts=10, rollouts=10):
        super().__init__(f'MonteCarlo::{name}', handouts, rollouts)
        self.handouts = handouts
        self.rollouts = rollouts

    def filter(self, valids):
        data = {}
        selector = selector_generator(data, self.valid_moves()) # //TODO: Create a real selector
        encoder = encoder_generator(self.max_number)
        rollout = rollout_maker(data)

        _, action, *_ = monte_carlo(
            self, 
            encoder, 
            rollout, 
            selector,
            self.handouts,
            self.rollouts,
        )

        return [action]
