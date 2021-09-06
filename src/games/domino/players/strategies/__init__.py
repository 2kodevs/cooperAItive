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

from .utils import state_to_list, selector_maker, encoder_generator, rollout_maker, monte_carlo, get_valids_data

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
]
