from ..player import BasePlayer
from .simpleh import SimpleHybrid
from .simple import BigDrop, Frequent
from .data_dropper import DataDropper
from .agachao import Agachao
from .repeater import Repeater
from .always_double import AlwaysDouble
from .table_counter import TableCounter
from .passer import Passer
import numpy as np


strategies        = [SimpleHybrid, BigDrop, DataDropper, Agachao, Repeater, AlwaysDouble, TableCounter, Frequent, Passer]
acummulated_value = [        0.26,     0.2,        0.16,    0.13,     0.09,         0.07,         0.05,     0.03,   0.01]


class Heuristic(*strategies):
    def __init__(self, name):
        self.coef = SimpleHybrid.PARAMETERS
        BasePlayer.__init__(self, f'Heuristic::{name}')

    def filter(self, valids):
        x = np.random.choice(list(range(9)), p=acummulated_value)
        return strategies[x].filter(self, valids)
