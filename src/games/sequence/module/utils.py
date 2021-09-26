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
        Check if the objects have the same color & sequence id

        * return False if sequence is None
        '''
        if not isinstance(other, Color):
            return False
        if other.bypass() or (None in [self.sequence, other.sequence]):
            return False
        return (self == other) and (self.sequence == other.sequence)

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
        return not self.color is None 

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
        super().__and__(other)
        return False


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


def get_rep(card):
    ctype, number = card
    return CARD_NUMBERS[number] + CARD_SIMBOLS[ctype.value]


def get_board_rep():
    return '\n'.join(str(', '.join(get_rep(x) for x in l)) for l in BOARD)


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
        