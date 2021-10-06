from random import shuffle
from .types import History, Card, List
from ...sequence import Event
from ..hands import split_cards, generate_cards


def get_discard_pile(history: History) -> List[Card]:
    pile = []
    for e, *data in history:
        if e in [Event.PLAY, Event.REMOVE, Event.DISCARD]:
            _, card, *_ = data
            pile.append(card)
        if e is Event.REFILL_DECK:
            pile = []
    return pile
    

def fixed_hand(cards, pile, id):
    taken = [*cards, *pile]
    taken.sort()
    full_deck = generate_cards()
    full_deck.sort()

    deck = []
    deck_iterator = iter(full_deck)
    taken_iterator = iter(taken)
    try:
        cur_card = next(deck_iterator)
        while True:
            cur_target = next(taken_iterator)
            while cur_card < cur_target:
                deck.append(cur_card)
                cur_card = next(deck_iterator)
            if cur_card == cur_target:
                cur_card = next(deck_iterator) 
    except StopIteration:
        deck.extend(deck_iterator)
    
    def hand(number_of_players, number_of_cards):
        shuffle(deck)

        player_first_card = number_of_cards * id
        all_cards = [*deck[:player_first_card], *cards, *deck[player_first_card:]]

        return split_cards(all_cards, number_of_players, number_of_cards)

    return hand
    