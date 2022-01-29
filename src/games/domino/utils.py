from module import get_player, get_hand
import re


def player_wrapper(player, *args):
    def player_builder(name):
        return player(name, *args)
    return player_builder


def prepare_player(data):
    player_name, *args = data
    player_class = get_player(player_name)
    return player_wrapper(player_class, *args)


def parse_hand(hand):
    r = r'\(([ \d]*),([ \d]*)\)'
    pieces = re.findall(r, hand)
    assert len(pieces) == 10, "Unable to parse the hand. Less than 10 pieces pattern (h1, h2) detected"
    data = []
    for x, y in pieces:
        a, b =  int(x), int(y)
        data.append((min(a, b), max(a, b)))
    return data
    