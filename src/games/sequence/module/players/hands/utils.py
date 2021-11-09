from ..player_view import PlayerView
from ...defaults import Card
from random import shuffle


def take(iterator, size):
    for _ in range(size):
        yield next(iterator)


def generate_cards():
    # Generate the cards
    cards = []
    for ctype in range(4):
        for num in range(1, 14):
            cards.append((Card(ctype), num))
    cards = [*cards, *cards]
    shuffle(cards)
    return cards


def split_cards(cards, number_of_player, cards_per_player):
    iterator = iter(cards)
    hands = [PlayerView(list(take(iterator, cards_per_player))) for _ in range(number_of_player)]
    return hands, list(iterator)
