from player import BasePlayer


class MCPlayer(BasePlayer):
    def __init__(self, name, handouts=10, rollouts=10):
        super().__init__(f'MonteCarlo::{name}')
        self.set_simulations(handouts, rollouts)

    def set_simulations(self, handouts, rollouts):
        self.handouts = handouts
        self.rollouts = rollouts


class MCSimulator(BasePlayer):
    def __init__(self, name):
        super().__init__(f'MonteCarloSimulator::{name}')
        self.states = [] # Add and state for each filter call

    def reset(self, position, pieces):
        for i in range(len(pieces)):
            a, b = pieces[i]
            pieces[i] = (min(a, b), max(a, b))
        pieces.sort()
        super().reset(position, pieces)

    def encode_game(self):
        ''' Return the current game state encoding
        '''
        raise NotImplementedError()

    def utility_value(self, data):
        ''' Return the utility value of the given data
        data: Data asociated to an <state, action> pair
        '''
        raise NotImplementedError()

    def get_utility(self):
        ''' Return the current state utility vector
        '''
        raise NotImplementedError()
