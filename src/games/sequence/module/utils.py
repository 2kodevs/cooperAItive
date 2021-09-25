from random import shuffle
from typing import Sequence


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

    @staticmethod
    def fixed(self):
        return not self.sequence is None 
    
    def __bool__(self):
        # return is the color is not None
        return not self.color is None 


class ByPassColor(Color):
    def bypass(self):
        return True

    def __and__(self, other):
        return isinstance(other, Color)

    def __eq__(self, other):
        super().__and__(other)
        return False


def append_type(deck, dtype):
    return [(x, dtype) for x in deck]


def hand_out(min_number=1, max_number=13):
    deck = [x for x in range(min_number, max_number + 1)]
    sequence_deck = []
    for t in range(4):
        sequence_deck.extend(append_type(deck, t))
    sequence_deck = [*sequence_deck, *sequence_deck]
    shuffle(sequence_deck)
    return sequence_deck

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