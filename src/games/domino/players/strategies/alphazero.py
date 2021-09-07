from ..player import BasePlayer
from .utils.alphazero import encoder_generator, rollout_maker, selector_maker
from .utils.mc import monte_carlo
from .models import alpha_zero_net as net
import torch

class AlphaZero(BasePlayer):
    def __init__(self, name, handouts=10, rollouts=50, NN=None):
        super().__init__(f'AlphaZero::{name}')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.NN = net(device=device)
        self.handouts = handouts
        self.rollouts = rollouts

    def filter(self, valids):
        data = {}
        selector = selector_maker(data, self.valid_moves(), self.pieces_per_player - len(self.pieces), False, 6)
        encoder = encoder_generator(self.max_number)
        rollout = rollout_maker(data, self.NN)

        _, action, *_ = monte_carlo(
            self, 
            encoder, 
            rollout, 
            selector,
            self.handouts,
            self.rollouts,
        )

        return [action]
