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
    

def calc_colab(sequence: Sequence, player: int):
    history = sequence.logs

    board = [[Color() for _ in range(len(l))] for l in BOARD]
    score = 0
    colors = set(sequence.colors)

    for e, *details in history:
        if e is Event.PLAY:
            playerId, _, color, (x, y) = details
            same_color_lines = lines_collector(board, color, x, y)
            other_color_lines = [
                lines_collector(board, color, x, y) 
                for color in colors if color != sequence.colors[playerId]
            ]
            board[x][y] = Color(color)
            # //TODO: do something with the lines
        elif e is Event.REMOVE:
            playerId, _, (x, y) = details
            if board[x][y].color == sequence.colors[playerId]:
                score -= 5 # //TODO: Add high penalization
                continue
            board[x][y] = Color()
            other_color_lines = [
                lines_collector(board, color, x, y) 
                for color in colors if color != sequence.colors[playerId]
            ]
            # //TODO: do something with the lines.
        elif e is Event.SEQUENCE:
            playerId, color, size = details
            if player == playerId:
                score += 1 + (size > 5) # //TODO: Add points for making a sequence
    return score # //TODO: Normalize
