from random import shuffle
from .types import History, Card, List, Sequence, Event
from ..hands import split_cards, generate_cards
from ...utils import BOARD, Color, lines_collector


def get_discard_pile(history: History) -> List[Card]:
    pile = []
    for e, *data in history:
        if e in [Event.PLAY, Event.REMOVE, Event.DISCARD]:
            _, card, *_ = data
            pile.append(card)
        if e is Event.REFILL_DECK:
            pile = []
    return pile
    

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
    

def calc_colab(sequence: Sequence, player: int):
    history = sequence.logs

    board = [[Color() for _ in range(len(l))] for l in BOARD]
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
                        board[x][y] = Color(other_color)
                        other_color_lines = lines_collector(board, other_color, x, y) 
                        score += lines_score(other_color_lines)
                        score_updates += 1

            # Execute the movement
            board[x][y] = Color(color)

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
                        board[x][y] = Color(other_color)
                        other_color_lines = lines_collector(board, other_color, x, y) 
                        score += lines_score(other_color_lines)     
                        score_updates += 1               
            board[x][y] = Color()
            
    return score / (200 * score_updates)
