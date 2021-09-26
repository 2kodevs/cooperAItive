from ..players import PlayerView
from ..defaults import Card
from .utils import take
from random import shuffle

def handout(number_of_player, pieces_per_player):
    # Check valid game distribution
    assert 2 * 4 * 13 >= number_of_player * pieces_per_player, "Not enough cards for this game"

    # Generate the cards
    cards = []
    for ctype in range(4):
        for num in range(1, 14):
            cards.append((Card(ctype), num))
    cards = [*cards, *cards]
    shuffle(cards)

    # handout the player cards and the deck 
    iterator = iter(cards)
    hands = [PlayerView(list(take(iterator, pieces_per_player))) for _ in range(number_of_player)]
    return hands, list(iterator)
