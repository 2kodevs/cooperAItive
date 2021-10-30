from random import shuffle
from .defaults import BOARD, CARD_SIMBOLS, CARD_NUMBERS


class Color:
    def __init__(self, color=None, seq=None):
        self.color = color
        self.sequence = seq

    def clone(self):
        return Color(self.color, self.sequence)

    def set_sequence(self, number):
        self.sequence = number

    def bypass(self):
        return False

    def __eq__(self, other):
        '''
        Check if the objects have the same sequence id

        * return False if sequence is None
        '''
        if not isinstance(other, Color):
            return False
        if other.bypass() or (None in [self.sequence, other.sequence]):
            return False
        return self.sequence == other.sequence

    def __and__(self, other):
        '''
        Check if the objects have the same color
        '''
        if not isinstance(other, Color):
            return False
        return (self.color == other.color) or other.bypass()

    @property
    def fixed(self):
        return self.sequence is not None 
    
    def __bool__(self):
        # return is the color is not None
        return self.color is not None 

    def __str__(self):
        if self.color is None:
            return " "
        return str(self.color)

    def __repr__(self):
        return str(self)


class ByPassColor(Color):
    def bypass(self):
        return True

    def __and__(self, other):
        return isinstance(other, Color)

    def __eq__(self, other):
        return False

    @property
    def fixed(self):
        return True


def find_pairs(board):
    pairs = {}

    for i, row in enumerate(board):
        for j, tile in enumerate(row):
            if tile not in pairs:
                pairs[tile] = []
            pairs[tile].append((i, j))

    return pairs


def printer(cards):
    output = [
        f'\t(Card.{card.name}, {num}): ' + str(value) 
        for (card, num), value in cards.items()
    ]
    print('{')
    print(',\n'.join(output))
    print('}')


BLACK   = "\x1b[30m"
RED     = "\x1b[31m"
GREEN   = "\x1b[32m"
YELLOW  = "\x1b[33m"
BLUE    = "\x1b[34m"
MAGENTA = "\x1b[35m"
CYAN    = "\x1b[36m"
RESET   = "\x1b[0m"
BOLD    = "\x1b[1m"
BLACKB  =  BLACK   +   BOLD
REDB    =  RED     +   BOLD
GREENB  =  GREEN   +   BOLD
YELLOWB =  YELLOW  +   BOLD
BLUEB   =  BLUE    +   BOLD
MAGENTAB=  MAGENTA +   BOLD
CYANB   =  CYAN    +   BOLD
COLORS = {
    None:RESET,
    'X':RESET,
    '0':RED, 
    '1':BLUE, 
    '2':GREEN, 
    '3':YELLOW,
}


def get_rep(card):
    ctype, number = card
    return CARD_NUMBERS[number] + CARD_SIMBOLS[ctype.value]


def get_color(color):
    if color is None: return " "
    try: return COLORS[color] + str(color) + RESET
    except: return str(color)


def get_piece_color(piece):
    return get_color(piece.color) + " X"[piece.fixed]


def get_board_cards_rep():
    return '\n'.join((', '.join(get_rep(x) for x in l) for l in BOARD))


def get_board_rep(board):
    it = iter(board)
    return '\n'.join(
        (', '.join(f'{get_rep(ca)}{get_piece_color(co)}' for ca, (_, co) in zip(cards, it))) 
        for cards in BOARD
    )


class BoardViewer:
    def __init__(self, board):
        self.board = board

    def __getitem__(self, pos):
        i, j = pos
        return super().__getattribute__('board')[i][j].clone()

    def __iter__(self):
        board = super().__getattribute__('board')
        for i, row in enumerate(board):
            for j, color in enumerate(row):
                yield (i, j), color.clone()

    def __getattribute__(self, name: str):
        raise AttributeError("BoardViewer doesn't have attributes")
        