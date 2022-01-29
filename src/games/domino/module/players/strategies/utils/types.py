from ...player_view import PlayerView
from typing import List, Any, Tuple, Callable, Dict
from ....domino import Domino, Event

State = int
Head = int
History = List[Any]
Piece = Tuple[int, int]
Action = Tuple[Piece, Head]
Encoder = Callable[[List[Piece], History, int], State]
RolloutMaker = Callable[[Domino, Encoder], None]
Selector = Callable[[State], List[Any]]
