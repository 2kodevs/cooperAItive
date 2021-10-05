from random import shuffle
from .types import History, Card, List
from ...sequence import Event
from ...hands import generate_cards, split_cards


def get_discard_pile(history: History) -> List[Card]:
    pile = []
    for e, *data in history:
        if e in [Event.PLAY, Event.REMOVE, Event.DISCARD]:
            card, *_ = data
            pile.append(card)
        if e is Event.REFILL_DECK:
            pile = []
    return pile
    

def fixed_hand(cards, pile, id):
    taken = set().union(cards()).union(pile)
    deck = list(set().union(generate_cards()).difference(taken))
    
    def hand(number_of_players, number_of_cards):
        shuffle(deck)

        player_first_card = number_of_cards * id + 1
        all_cards = [*deck[:player_first_card], *cards(), *deck[player_first_card:]]

        return split_cards(all_cards, number_of_players, number_of_cards)

    return hand
    