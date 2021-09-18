from random import shuffle

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