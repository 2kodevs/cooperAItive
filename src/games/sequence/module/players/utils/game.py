from random import shuffle

from ...defaults import ALL_CARDS_MAPPING, CORNERS, JACK
from .types import Card, GameData, List, Position, Sequence, Event, Action, State
from ..hands import split_cards, generate_cards
from ...utils import BOARD, BoardViewer, Piece, lines_collector, take


def order_hand(cards, pile, id, number_of_cards):
    taken = [*cards, *pile]
    taken.sort()
    full_deck = generate_cards()
    full_deck.sort()

    deck = []
    deck_iterator = iter(full_deck)
    taken_iterator = iter(taken)
    try:
        deck.append(next(deck_iterator))
        while True:
            cur_target = next(taken_iterator)
            while deck[-1] < cur_target:
                deck.append(next(deck_iterator))
            if deck[-1] == cur_target:
                deck.pop()
                deck.append(next(deck_iterator))
    except StopIteration:
        deck.extend(deck_iterator)
    
    shuffle(deck)

    player_first_card = number_of_cards * id
    all_cards = [*deck[:player_first_card], *cards, *deck[player_first_card:]]

    return all_cards


def fixed_hand(cards, pile, id, number_of_cards):
    order = order_hand(cards, pile, id, number_of_cards)

    def hand(number_of_players, number_of_cards):
        return split_cards(order, number_of_players, number_of_cards)

    return hand


def lines_score(lines):
    size = [0] * 6
    for line in lines:
        sub0, sub1 = len(line[:5]), len(line[5:])
        size[sub0] += 1
        size[sub1] += 1
    score = 0
    for i, cant in enumerate(size):
        score += i * i * cant
    return score
    

def move_score(*collector_params):
    lines = lines_collector(*collector_params)
    return lines_score(lines)


def calc_colab(sequence: Sequence, player: int):
    history = sequence.logs

    board = [[Piece() for _ in range(len(l))] for l in BOARD]
    score = 0
    colors = set(sequence.colors)
    seq_id = 0
    score_updates = 0

    for e, *details in history:
        if e is Event.PLAY:
            playerId, _, color, (x, y) = details

            if playerId == player: 
                # add score per damage
                for other_color in colors:
                    if other_color != sequence.colors[playerId]:
                        board[x][y] = Piece(other_color)
                        score += move_score(board, other_color, x, y) 
                        score_updates += 1

            # Execute the movement
            board[x][y] = Piece(color)

            same_color_lines = lines_collector(board, color, x, y)
            if playerId == player: 
                # add team movement score
                score += lines_score(same_color_lines)
                score_updates += 1

            # Set sequences
            for line in same_color_lines:
                size = len(line)
                seq = [0, 5, 9][(size >= 5) + (size >= 9)]
                if seq:
                    for i, j in line[:size]:
                        board[i][j].set_sequence(seq_id)
                    seq_id += 1

        elif e is Event.REMOVE:
            playerId, _, (x, y) = details
            if playerId == player: 
                # add score per damage
                for other_color in colors:
                    if other_color != sequence.colors[playerId]:
                        board[x][y] = Piece(other_color)
                        score += move_score(board, other_color, x, y) 
                        score_updates += 1               
            board[x][y] = Piece()
            
    return score / (200 * score_updates)


def table_bit(i: int, j: int) -> int:
    return i * 10 + j


def encode_board(board, color, all_colors):
    masks = {c:0 for c in all_colors}
    for pos, piece in board:
        if piece and not piece.bypass():
            masks[piece.color] |= (1 << table_bit(*pos))
    return masks.pop(color), masks


def encode_cards(cards: List[Card]) -> int:
    mask = 0
    data = {c:0 for c in cards}
    for c in cards:
        mask |= (1 << table_bit(*ALL_CARDS_MAPPING[c][data[c]]))
        data[c] += 1
    return mask


def adjust_shifting(pos: Position) -> int:
    return len([1 for x in CORNERS if x < pos])


def encode_valids(valids: List[Action]) -> int:
    if valids[0] is None:
        return 1 << 198
    mask = 0
    discards = 0
    vector = []
    for (_, num), pos in valids:
        if pos is None:
            mask |= (1 << (192 + discards))
            vector.append(192 + discards)
            discards += 1
        else:
            cur_bit = 1 << (table_bit(*pos) - adjust_shifting(pos))
            if num is JACK: 
                temp1 = mask                
                mask |= (cur_bit << 96)
                vector.append((table_bit(*pos) - adjust_shifting(pos)) + 96)
                assert temp1 != mask, 'JACK \n' + str(valids) +  '\n' + repr(vector) + '\n' + str(mask) + '\n' + str(temp1)
            else:
                temp1 = mask  
                mask |= cur_bit
                vector.append((table_bit(*pos) - adjust_shifting(pos)))
                assert temp1 != mask, 'No JACK \n' + str(valids) + '\n' + repr(vector) + '\n' + str(mask) + '\n' + str(temp1)
    if discards:
        print(discards)
    return mask


def encode(
    player: GameData,
    discard_pile: List[Card],
) -> State :       
    player_board, boards = encode_board(
        player.board, 
        player.color,
        player.colors,
    )
    cards = encode_cards(list(player.cards))
    pile = encode_cards(discard_pile)
    offset = 0
    state = 0
    for mask in [player_board, *boards.values(), cards, pile]:
        state += (mask << offset)
        offset += 110 # 11 x 10 state boards
    state += (player.can_discard << (offset - 1))
    return state
   

def state_to_list(
    state: State,
    size: int,
) -> List[int]:
    binary_rep = bin(state)[2:]
    binary_rep = '0' * max(0, size - len(binary_rep)) + binary_rep
    return [int(x) for x in binary_rep[-1 : -(size + 1) : -1]]


def split_list(l, rows, cols):
    it = iter(l)
    return [list(take(it, cols)) for _ in range(rows)]


def state_number_to_matrix(state: int, number_of_matrixes: int = 4):
    state_list = state_to_list(state, number_of_matrixes * 110)
    matrix_data = split_list(state_list, number_of_matrixes, 110)
    return [split_list(l, 11, 10) for l in matrix_data]
    