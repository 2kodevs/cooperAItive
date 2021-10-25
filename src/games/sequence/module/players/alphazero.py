from .player import BasePlayer
from .utils.alphazero import encode, rollout_maker, selector_maker
from .utils.mc import monte_carlo
from .models import AlphaZeroNet

class AlphaZero(BasePlayer):
    def __init__(self, name, handouts, rollouts, NN):
        super().__init__(f'AlphaZero::{name}')
        self.turn = 0

        if isinstance(NN, str):
            self.NN = AlphaZeroNet.load(NN)
        else: 
            self.NN = NN
        self.handouts = int(handouts)
        self.rollouts = int(rollouts)

    def step(self):
        self.turn += 1 # turns when a discard occur will be counted twice
        return super().step()

    def filter(self, valids):
        data = {}
        selector = selector_maker(data, self.valid_moves(), self.turn, False, 50) #//TODO: Change tau_threshold
        rollout = rollout_maker(data, self.NN)

        _, action, *_ = monte_carlo(
            self, 
            encode, 
            rollout, 
            selector,
            self.handouts,
            self.rollouts,
        )

        return [action]
