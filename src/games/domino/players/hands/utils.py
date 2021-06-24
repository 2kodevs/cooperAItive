from random import sample, randint

def share_data(max_number, pieces_per_player, high):
    data_num = randint([0, max_number // 2 + 1][high], [max_number // 2, max_number][high])
    data = [(min(data_num, i), max(data_num, i)) for i in sample(list(range(max_number + 1)), min(randint(max_number - 1, max_number + 1), pieces_per_player))]

    all_pieces = {(i, j) for i in range(max_number + 1) for j in range(max_number + 1) if i <= j}

    data0 = data[:randint(1, len(data))]
    pieces0 = all_pieces - set(data)
    hand0 = sample(pieces0, pieces_per_player - len(data0))
    pieces0 -= set(hand0)
    hand0 += data0

    data1 = set(data) - set(data0)
    pieces1 = pieces0 - set(data1)
    hand1 = sample(pieces1, pieces_per_player - len(data1))
    pieces1 -= set(hand1)
    hand1 += data1

    assert 4 * pieces_per_player <= len(pieces1) + len(hand1) + len(hand0)
    hand = sample(pieces1, 2 * pieces_per_player)
    temp_hands = [hand[i:i+pieces_per_player] for i in range(0, 2 * pieces_per_player, pieces_per_player)]

    return (hand0, hand1, temp_hands)
    