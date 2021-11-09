from random import shuffle
from .defaults import BOARD, CARD_SIMBOLS, CARD_NUMBERS


class Piece:
    def __init__(self, color=None, seq=None):
        self.color = color
        self.sequence = seq

    def clone(self):
        return Piece(self.color, self.sequence)

    def set_sequence(self, number):
        self.sequence = number

    def bypass(self):
        return False

    def __eq__(self, other):
        '''
        Check if the objects have the same sequence id

        * return False if sequence is None
        '''
        if not isinstance(other, Piece):
            return False
        if other.bypass() or (None in [self.sequence, other.sequence]):
            return False
        return self.sequence == other.sequence

    def __and__(self, other):
        '''
        Check if the objects have the same color
        '''
        if not isinstance(other, Piece):
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


class ByPassPiece(Piece):
    def bypass(self):
        return True

    def __and__(self, other):
        return isinstance(other, Piece)

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
            for j, piece in enumerate(row):
                yield (i, j), piece.clone()

    def __getattribute__(self, name: str):
        raise AttributeError("BoardViewer doesn't have attributes")
        

def lines_collector(board, color, i, j):
    # check for sequences
    data = [[], [], [], []] # one per direction
    moves = [
        # (i, j, data)
        (-1, -1, 0),
        (0, -1, 1),
        (1, -1, 2),
        (1, 0, 3),
    ]

    # check a half of the line
    for inc_i, inc_j, idx in moves:
        last = Piece(color)
        cur_i, cur_j = i, j
        while last & board[cur_i][cur_j]:
            if last == board[cur_i][cur_j]:
                break
            last = board[cur_i][cur_j]
            data[idx].append((cur_i, cur_j))
            cur_i += inc_i
            cur_j += inc_j
            if not ((0 <= cur_i < 10) and (0 <= cur_j < 10)):
                break
        data[idx] = data[idx][::-1]

    # check the other line half
    for inc_i, inc_j, idx in moves:
        last = Piece(color)
        cur_i, cur_j = i - inc_i, j - inc_j
        if not ((0 <= cur_i < 10) and (0 <= cur_j < 10)):
            continue
        while last & board[cur_i][cur_j]:
            if last == board[cur_i][cur_j]:
                break
            last = board[cur_i][cur_j]
            data[idx].append((cur_i, cur_j))
            cur_i -= inc_i
            cur_j -= inc_j
            if not ((0 <= cur_i < 10) and (0 <= cur_j < 10)):
                break

    return data
    