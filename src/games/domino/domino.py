import argparse
from utils import prepare_player as get_player
from module import get_rule, get_hand, PLAYERS, RULES, BEHAVIORS, HANDS, get_player as get_player_by_name


def info(args):
    information = \
            'Players:\n' + \
            ''.join([f'+ {player.__name__.lower()}\n' for player in PLAYERS]) + \
            '\n' + \
            'Rules:\n' + \
            ''.join([f'+ {rule.__name__.lower()}\n' for rule in RULES]) + \
            '\nOptionally you can merge some players into one passing any amount of them\n' + \
            'separated by a dash(-). Example: BigDrop-Repeater\n' + \
            'Also you can use some extra behaviors in your player mesures.\n' + \
            '\nAvailable behaviors:\n' + \
            ''.join([f'+ {rule.__name__.lower()}\n' for rule in BEHAVIORS]) + \
            '\n' + \
            'Hands:\n' + \
            ''.join([f'+ {hand.__name__.lower()}\n' for hand in HANDS])
    print(information)


def play(args):
    player0 = get_player(args.player0)
    player1 = get_player(args.player1)
    player2 = get_player(args.player0 if args.player2 is None else args.player2)
    player3 = get_player(args.player1 if args.player3 is None else args.player3)
    rule = get_rule(args.rule)
    hand = get_hand(args.hand)

    status = {-1:0, 0:0, 1:0}
    for _ in range(args.rep):
        game = rule()
        status[game.start(player0, player1, player2, player3, hand, *args.pieces)] += 1
    if args.verbose:
        print(status)
    return status


def match(args):
    player = get_player(args.player)
    rule = get_rule(args.rule)
    hand = get_hand(args.hand)

    status = {-1:0, 0:0, 1:0, 2:0}
    for _ in range(args.rep):
        for other in args.oponents:
            oponent = get_player_by_name(other)
            game = rule()
            status[game.start(player, oponent, player, oponent, hand, *args.pieces)] += 1
            game = rule()
            status[1 - game.start(oponent, player, oponent, player, hand, *args.pieces)] += 1
    status[-1] += status.pop(2)
    if args.verbose:
        print(status)
    return status 


def main():
    parser = argparse.ArgumentParser("DomAIno")

    subparsers = parser.add_subparsers()
    info_parser = subparsers.add_parser('info', help="Show available Players and Rules")
    info_parser.set_defaults(command=info)

    play_parser = subparsers.add_parser('play', help="Run a domino game")
    play_parser.add_argument('-p0',  '--player0',     dest='player0', nargs='+', default=['random'], help="Player0 class name & arguments if needed")
    play_parser.add_argument('-p1',  '--player1',     dest='player1', nargs='+', default=['random'], help="Player1 class name & arguments if needed")
    play_parser.add_argument('-p2',  '--player2',     dest='player2', nargs='+', default=None, help="Player2 class name & arguments if needed")
    play_parser.add_argument('-p3',  '--player3',     dest='player3', nargs='+', default=None, help="Player3 class name & arguments if needed")
    play_parser.add_argument('-r',   '--rule',        dest='rule',    default='onegame', help="Game rule to use in each play")
    play_parser.add_argument('-n',   '--nine',        dest='pieces',  action='store_const', const=[9,10], default=[], help="Double nine mode")
    play_parser.add_argument('-rep', '--repetitions', dest='rep',     type=int, default=1, help="Numbers of plays to run")
    play_parser.add_argument('-H',   '--hand',        dest='hand',    default='hand_out', help="Game handout strategy")
    play_parser.add_argument('-v',   '--verbose',     dest='verbose', action='store_true', help="Print the game result at the end")

    play_parser.set_defaults(command=play)

    match_parser = subparsers.add_parser('match', help="Run a domino match")
    match_parser.add_argument('-p',    '--player',     dest='player',   nargs='+', default=['random'], help="Player class name & arguments if needed")
    match_parser.add_argument('-r',   '--rule',        dest='rule',     default='onegame', help="Game rule to use in each play")
    match_parser.add_argument('-n',   '--nine',        dest='pieces',   action='store_const', const=[9,10], default=[], help="Double nine mode")
    match_parser.add_argument('-rep', '--repetitions', dest='rep',      type=int, default=1, help="Numbers of plays to run per oponent")
    match_parser.add_argument('-H',   '--hand',        dest='hand',     default='hand_out', help="Game handout strategy")
    match_parser.add_argument('-o',   '--oponents',    dest='oponents', type=str, nargs='+', required=True, help="Oponents class names")
    match_parser.add_argument('-v',   '--verbose',     dest='verbose', action='store_true', help="Print the game result at the end")
    match_parser.set_defaults(command=match)

    args = parser.parse_args()

    if not hasattr(args, 'command'):
        parser.print_help()
    else:
        args.command(args)


if __name__ == '__main__':
    main()
