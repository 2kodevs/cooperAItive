from .utils import split_cards, generate_cards

def handout(number_of_player, pieces_per_player):
    # Check valid game distribution
    assert 2 * 4 * 13 >= number_of_player * pieces_per_player, "Not enough cards for this game"

    cards = generate_cards()

    return split_cards(cards, number_of_player, pieces_per_player)
