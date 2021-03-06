from typing import List, Any, Tuple, Callable, Dict
from ...defaults import Card as CardType
from ...sequence import Sequence, Event


class GameData:
    @property
    def color(self): pass

    @property
    def colors(self): pass
    
    @property
    def cards(self): pass

    @property
    def board(self): pass

    @property
    def can_discard(self): pass


State = int
History = List[Any]
Card = Tuple[CardType, int]
Position = Tuple[int, int]
Action = Tuple[Card, Position]
Encoder = Callable[[GameData, List[Card]], State]
RolloutMaker = Callable[[Sequence, Encoder], None]
Selector = Callable[[State], List[Any]]
