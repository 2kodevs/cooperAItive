from ..player import BasePlayer
from random import random

class RLPlayer(BasePlayer):
    ''' 
    This player use reinforcement learning
    to measure his strategy.
    '''
    def __init__(self, name):
        super().__init__(f"RLPlayer::{name}")
        self.actions = []
        self.values = {}
        self.steps = 0

    def load_values(self, addr):
        pass
    
    def save_values(self, addr):
        pass

    def filter(self, valids=None):
        return self.policy(super().filter(valids))

    def get_movements(self):
        heads = [-1, -1]
        first_move = True

        self.movements = [[[0, 0] for _ in range(10)] for _ in range(4)]

        for e, *d in self.history:
            if e.name == 'MOVE':
                player, piece, head = d
                self.movements[player][piece[0]][piece[0] == heads[head]] += 1
                self.movements[player][piece[1]][piece[1] == heads[head]] += 1
                if first_move:
                    heads = list(piece)
                    first_move = False
                else:
                    heads[head] = piece[piece[0] == heads[head]]

    def get_rep(self, piece, head):
        '''
        Build a action.
        '''
        piece = list(piece)
        if piece[0] != self.heads[head]:
            piece.reverse()
        rep = []
        for num in piece:
            # cur = []
            # for player in range(4):
            #     cur.extend(self.movements[player][num])
            # for x in range(self.me * 2):
            #     cur.append(cur[x])
            # add playe movements related to num
            for i in range(4):
                rep.extend(self.movements[(self.me + i) % 4][num])
            # rep.extend(cur[self.me * 2:])
            data = 0
            for p in self.pieces:
                data += (num in p)
            # add num data size (bit here :-P)
            rep.append(data)
        # add game stage (bit here :-P)
        rep.append((self.steps + 9) // 10)
        return str(rep)

    #//TODO: Infer value here from NN
    def get_value(self, piece, head):
        return self.values.get(self.get_rep(piece, head), [0.5, 0])[0]

    def policy(self, valids):
        #//TODO: Parametrize exploration constant
        if random() <= 0.3: return valids
        self.get_movements()

        top = 0
        greedy = []
        for action in valids: 
            value = self.get_value(*action)
            if value > top:
                greedy.clear()
                top = value
            if value == top: greedy.append(action)
        return greedy

    def log(self, data):
        super().log(data)
        e, *d = data
        if e.name == 'MOVE':
            self.steps += 1
            player, piece, head = d
            if player == self.me:
                self.actions.append(self.get_rep(piece, head))
        elif e.name == 'PASS':
            self.steps += 1
        elif e.name == 'WIN':
            new_val = [0, 1][d[0] == self.me % 2]
            self.measure(new_val)

    def measure(self, new_val):
        #//TODO: Parametrize step_size_numerator
        step_size_numerator = 0.1

        for action in self.actions:
            value, n = self.values.get(action, [0.5, 0])
            step_size = step_size_numerator / (n + 1)
            final_value = value + step_size * (new_val - value)
            self.values[action] = [final_value, n + 1]
        self.actions.clear()


class SingletonRLPlayer:
    def __init__(self, *args):
        self.init = args
        self.instances = {}
        self.instances["0"] = RLPlayer("0")
        self.instances["2"] = RLPlayer("2")
        self.instances["1"] = self.instances["0"]
        self.instances["3"] = self.instances["2"]

        self.__name__ = 'RLPlayer'

    def __call__(self, name):
        if not name in self.instances:
            print("wtf")
            self.instances[name] = RLPlayer(name)
        return self.instances[name]
