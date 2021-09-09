from module import get_player


def player_wrapper(player, *args):
    def player_builder(name):
        return player(name, *args)
    return player_builder


def prepare_player(data):
    player_name, *args = data
    player_class = get_player(player_name)
    return player_wrapper(player_class, *args)
    