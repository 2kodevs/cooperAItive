from enum import Enum


JACK = 11
CARD_SIMBOLS = ["❤︎", "♦", "♠", "♣"]


class Card(Enum):
    HEART = 0     # ❤︎
    DIAMOND = 1   # ♦
    SPADES = 2    # ♠
    CLUBS = 3     # ♣
    ESPECIAL = 4  # Board corners


REMOVE = [Card.CLUBS, Card.SPADES] # //TODO: Select the correct card types


# board from https://steamcommunity.com/sharedfiles/filedetails/?id=639670109&searchtext=
BOARD = [
    [(Card.ESPECIAL, 0), (Card.DIAMOND, 6), (Card.DIAMOND, 7) , (Card.DIAMOND, 8), (Card.DIAMOND, 9), (Card.DIAMOND, 10), (Card.DIAMOND, 12), (Card.DIAMOND, 13), (Card.DIAMOND, 1), (Card.ESPECIAL, 0)],
    [(Card.DIAMOND, 5) , (Card.HEART, 3)  , (Card.HEART, 2)   , (Card.SPADES, 2) , (Card.SPADES, 3) , (Card.SPADES, 4)  , (Card.SPADES, 5)  , (Card.SPADES, 6)  , (Card.SPADES, 7) , (Card.CLUBS, 1)   ],
    [(Card.DIAMOND, 4) , (Card.HEART, 4)  , (Card.DIAMOND, 13), (Card.DIAMOND, 1), (Card.CLUBS, 1)  , (Card.CLUBS, 13)  , (Card.CLUBS, 12)  , (Card.CLUBS, 10)  , (Card.SPADES, 8) , (Card.CLUBS, 13)  ],
    [(Card.DIAMOND, 3) , (Card.HEART, 5)  , (Card.DIAMOND, 12), (Card.HEART, 12) , (Card.HEART, 10) , (Card.HEART, 9)   , (Card.HEART, 8)   , (Card.CLUBS, 9)   , (Card.SPADES, 9) , (Card.CLUBS, 12)  ],
    [(Card.DIAMOND, 2) , (Card.HEART, 6)  , (Card.DIAMOND, 10), (Card.HEART, 13 ), (Card.HEART, 3)  , (Card.HEART, 2)   , (Card.HEART, 7)   , (Card.CLUBS, 8)   , (Card.SPADES, 10), (Card.CLUBS, 10)  ],
    [(Card.SPADES, 1)  , (Card.HEART, 7)  , (Card.DIAMOND, 9) , (Card.HEART, 1)  , (Card.HEART, 4)  , (Card.HEART, 5)   , (Card.HEART, 6)   , (Card.CLUBS, 7)   , (Card.SPADES, 12), (Card.CLUBS, 9)   ],
    [(Card.SPADES, 13) , (Card.HEART, 8)  , (Card.DIAMOND, 8) , (Card.CLUBS, 2)  , (Card.CLUBS, 3)  , (Card.CLUBS, 4)   , (Card.CLUBS, 5)   , (Card.CLUBS, 6)   , (Card.SPADES, 13), (Card.CLUBS, 8)   ],
    [(Card.SPADES, 12) , (Card.HEART, 9)  , (Card.DIAMOND, 7) , (Card.DIAMOND, 6), (Card.DIAMOND, 5), (Card.DIAMOND, 4) , (Card.DIAMOND, 3) , (Card.DIAMOND, 2) , (Card.SPADES, 1) , (Card.CLUBS, 7)   ],
    [(Card.SPADES, 10) , (Card.HEART, 10) , (Card.HEART, 12)  , (Card.HEART, 13) , (Card.HEART, 1)  , (Card.CLUBS, 2)   , (Card.CLUBS, 3)   , (Card.CLUBS, 4)   , (Card.CLUBS, 5)  , (Card.CLUBS, 6)   ],
    [(Card.ESPECIAL, 1), (Card.SPADES, 9) , (Card.SPADES, 8)  , (Card.SPADES, 7) , (Card.SPADES, 6) , (Card.SPADES, 5)  , (Card.SPADES, 4)  , (Card.SPADES, 3)  , (Card.SPADES, 2) , (Card.ESPECIAL, 1)],
]


CARDS_POSITIONS = {
    (Card.DIAMOND, 6):  [(0, 1), (7, 3)],
    (Card.DIAMOND, 7):  [(0, 2), (7, 2)],
    (Card.DIAMOND, 8):  [(0, 3), (6, 2)],
    (Card.DIAMOND, 9):  [(0, 4), (5, 2)],
    (Card.DIAMOND, 10): [(0, 5), (4, 2)],
    (Card.DIAMOND, 12): [(0, 6), (3, 2)],
    (Card.DIAMOND, 13): [(0, 7), (2, 2)],
    (Card.DIAMOND, 1):  [(0, 8), (2, 3)],
    (Card.DIAMOND, 5):  [(1, 0), (7, 4)],
    (Card.HEART, 3):    [(1, 1), (4, 4)],
    (Card.HEART, 2):    [(1, 2), (4, 5)],
    (Card.SPADES, 2):   [(1, 3), (9, 8)],
    (Card.SPADES, 3):   [(1, 4), (9, 7)],
    (Card.SPADES, 4):   [(1, 5), (9, 6)],
    (Card.SPADES, 5):   [(1, 6), (9, 5)],
    (Card.SPADES, 6):   [(1, 7), (9, 4)],
    (Card.SPADES, 7):   [(1, 8), (9, 3)],
    (Card.CLUBS, 1):    [(1, 9), (2, 4)],
    (Card.DIAMOND, 4):  [(2, 0), (7, 5)],
    (Card.HEART, 4):    [(2, 1), (5, 4)],
    (Card.CLUBS, 13):   [(2, 5), (2, 9)],
    (Card.CLUBS, 12):   [(2, 6), (3, 9)],
    (Card.CLUBS, 10):   [(2, 7), (4, 9)],
    (Card.SPADES, 8):   [(2, 8), (9, 2)],
    (Card.DIAMOND, 3):  [(3, 0), (7, 6)],
    (Card.HEART, 5):    [(3, 1), (5, 5)],
    (Card.HEART, 12):   [(3, 3), (8, 2)],
    (Card.HEART, 10):   [(3, 4), (8, 1)],
    (Card.HEART, 8):    [(3, 6), (6, 1)],
    (Card.HEART, 9):    [(3, 5), (7, 1)],
    (Card.CLUBS, 9):    [(3, 7), (5, 9)],
    (Card.SPADES, 9):   [(3, 8), (9, 1)],
    (Card.DIAMOND, 2):  [(4, 0), (7, 7)],
    (Card.HEART, 6):    [(4, 1), (5, 6)],
    (Card.HEART, 13):   [(4, 3), (8, 3)],
    (Card.HEART, 7):    [(4, 6), (5, 1)],
    (Card.CLUBS, 8):    [(4, 7), (6, 9)],
    (Card.SPADES, 10):  [(4, 8), (8, 0)],
    (Card.SPADES, 1):   [(5, 0), (7, 8)],
    (Card.HEART, 1):    [(5, 3), (8, 4)],
    (Card.CLUBS, 7):    [(5, 7), (7, 9)],
    (Card.SPADES, 12):  [(5, 8), (7, 0)],
    (Card.SPADES, 13):  [(6, 0), (6, 8)],
    (Card.CLUBS, 2):    [(6, 3), (8, 5)],
    (Card.CLUBS, 3):    [(6, 4), (8, 6)],
    (Card.CLUBS, 4):    [(6, 5), (8, 7)],
    (Card.CLUBS, 5):    [(6, 6), (8, 8)],
    (Card.CLUBS, 6):    [(6, 7), (8, 9)],
}


CORNERS = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
