from typing import List, Any, Tuple, Callable, Dict
from ....domino import Domino

State = int
Head = int
History = List[Any]
Piece = Tuple[int, int]
Action = Tuple[Piece, Head]
Encoder = Callable[[List[Piece], History, int], State]
RolloutMaker = Callable[[Domino, Encoder]]
Selector = Callable[[State], Action]
