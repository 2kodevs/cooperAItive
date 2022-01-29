from .simple import BigDrop, Frequent, Random
from .rlplayer import SingletonRLPlayer
from .table_counter import TableCounter
from .always_double import AlwaysDouble
from .player_merge import MergeFactory
from .data_dropper import DataDropper
from .less_played import LessPlayed
from .data_keeper import DataKeeper
from .supportive import Supportive
from .small_drop import SmallDrop
from .simpleh import SimpleHybrid
from .double_end import DoubleEnd
from .non_double import NonDouble
from .repeater import Repeater
from .agachao import Agachao
from .passer import Passer
from .alphazero import AlphaZero
from .monte_carlo import MonteCarlo
from .human import Human
from .heuristic import Heuristic
from .remote import Remote

from .utils import alphazero as alphazero_utils, mc as mc_utils, game as game_utils
from .models import alpha_zero_model as AlphaZeroModel, AlphaZeroNet


class Shortcut:
    def __init__(self, name, cls) -> None:
        self.__name__ = name
        self.cls = cls

    def __call__(self, *args, **kwds):
        return self.cls(*args, **kwds)


# Add players to this list
PLAYERS = [
    BigDrop,
    Frequent,
    Random,
    SimpleHybrid,
    Repeater,
    TableCounter,
    Passer,
    Supportive,
    LessPlayed,
    DataKeeper,
    SmallDrop,
    Agachao,
    DataDropper,
    AlwaysDouble,
    DoubleEnd,
    NonDouble,
    SingletonRLPlayer(),
    Human,
    MonteCarlo,
    AlphaZero,
    Heuristic,
    Shortcut("MC", MonteCarlo),
    Shortcut("A0", AlphaZero),
    Shortcut("best", MergeFactory([
        SimpleHybrid, DataDropper, Agachao, 
        Repeater, AlwaysDouble, TableCounter, Passer,
    ])),
    Remote,
]
