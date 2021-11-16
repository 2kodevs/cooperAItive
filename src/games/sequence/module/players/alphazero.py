from .player import BasePlayer
from .utils.alphazero import rollout_maker, selector_maker
from .utils.game import encode
from .utils.mc import monte_carlo
from .models import AlphaZeroNet

class AlphaZero(BasePlayer):
    def __init__(self, name, handouts, rollouts, NN, coop = 1, cput = 1):
        super().__init__(f'AlphaZero::{name}')
        self.turn = 0

        if isinstance(NN, str):
            self.NN = AlphaZeroNet.load(NN)
        else: 
            self.NN = NN
        self.handouts = int(handouts)
        self.rollouts = int(rollouts)
        self.coop = int(coop)
        self.cput = int(cput)

    def step(self):
        self.turn += 1 # turns when a discard occur will be counted twice
        return super().step()

    def filter(self, valids):
        data = {}
        selector = selector_maker(data, self.valid_moves(), self.turn, False, 50) #//TODO: Change tau_threshold
        rollout = rollout_maker(data, self.NN, self.coop, self.cput)

        _, action, *_ = monte_carlo(
            self, 
            encode, 
            rollout, 
            selector,
            self.handouts,
            self.rollouts,
        )

        return [action]
