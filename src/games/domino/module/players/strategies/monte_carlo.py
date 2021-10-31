from ..player import BasePlayer
from .utils.alphazero import encoder_generator
from .utils.mc import monte_carlo, rollout_maker, selector_maker


class MonteCarlo(BasePlayer):
    def __init__(self, name, handouts=10, rollouts=50):
        super().__init__(f'MonteCarlo::{name}')
        self.handouts = int(handouts)
        self.rollouts = int(rollouts)

    def filter(self, valids):
        data = {}
        selector = selector_maker(data, self.valid_moves())
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
