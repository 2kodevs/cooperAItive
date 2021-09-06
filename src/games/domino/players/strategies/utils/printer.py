def piece_horizontal(piece):
    return "\n┏━━━┳━━━┓\n┃ %d ┃ %d ┃\n┗━━━┻━━━┛" % piece


def piece_vertical(piece):
    return "\n┏━━━┓\n┃ %d ┃\n┣━━━┫\n┃ %d ┃\n┗━━━┛" % piece


def take(iterator, size):
    a = 0
    for x in iterator:
        yield x
        a += 1
        if a == size:
            break
    # raise StopIteration()    


def infinite_list(iterator, element):
    for x in iterator:
        yield x
    while True:
        yield element


def game_view(history):
    heads = None
    initial = ""
    sides = ["", ""]
    for e, *details in history:
        if e.name == "MOVE":
            _, (a, b), head = details
            try:
                if heads[head] != [a, b][head]:
                    a, b = b, a
                if head:
                    a, b = b, a
                sides[head] += piece_vertical((a, b))
                heads[head] = [a, b][a == heads[head]]
            except TypeError:
                heads = [a, b]
                initial += piece_horizontal((a, b))
    data = [
        sides[0].split('\n'),
        initial.split('\n'),
        sides[1].split('\n'),
    ]
    size = max(len(x) for x in data)
    view = [infinite_list(x, " " * sz) for x, sz in zip(data, [5, 9, 5])]
    table = '\n'.join(x + y + z for x, y, z in take(zip(*view), size))
    return table, data


def game_printer(history):
    table, _ = game_view(history)
    print(table)


def hand_view(pieces):
    ordered_pieces = [(min(a, b), max(a, b)) for a, b in pieces]
    ordered_pieces.sort()
    data = [piece_vertical(p).split('\n') for p in ordered_pieces]
    
    hand = '\n'.join(
        ''.join(row) for row in zip(*data)   
    )
    return hand, data


def hand_printer(pieces):
    hand, _ = hand_view(pieces)
    print(hand)
