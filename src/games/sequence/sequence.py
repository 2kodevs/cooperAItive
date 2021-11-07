import argparse
from utils import prepare_player as get_player
from module import get_hand, PLAYERS, HANDS, SequenceManager, get_player as get_player_by_name


def info(args):
    information = ''.join([
        'Players:\n',
        ''.join([f'+ {player.__name__.lower()}\n' for player in PLAYERS]),
        '\nHands:\n',
        ''.join([f'+ {hand.__name__.lower()}\n' for hand in HANDS]),
    ])
    print(information)


def play(args):
    player0 = get_player(args.player0)
    player1 = get_player(args.player1)
    player2 = get_player(args.player0 if args.player2 is None else args.player2)
    player3 = get_player(args.player1 if args.player3 is None else args.player3)
    players = [player0, player1, player2, player3]
    hand    = get_hand(args.hand)
    colors  = args.colors
    status = {x:0 for x in colors}
    status[None] = 0

    manager = SequenceManager()
    
    for _ in range(args.rep):
        status[manager.run(hand, players, colors, args.cards, args.win)] += 1
    if args.verbose:
        print(status)
    return status


def match(args):
    player = get_player(args.player)
    hand   = get_hand(args.hand)
    c1, c2 = iter(args.colors)
    status1 = {c1:0, c2:0, None:0}
    status2 = {c1:0, c2:0, None:0}

    manager = SequenceManager()
    for _ in range(args.rep):
        for other in args.oponents:
            oponent = get_player_by_name(other)
            status1[manager.run(hand, [player, oponent, player, oponent], [c1, c2, c1, c2], args.cards, args.win)] += 1
            status2[manager.run(hand, [oponent, player, oponent, player], [c2, c1, c2, c1], args.cards, args.win)] += 1
    
    if args.verbose:
        print("player vs other ->", status1)
        print("other vs player ->", status2)
    return status1, status2


def main():
    parser = argparse.ArgumentParser("Sequence IA")

    subparsers = parser.add_subparsers()
    info_parser = subparsers.add_parser('info', help="Show available Players and Hands")
    info_parser.set_defaults(command=info)

    play_parser = subparsers.add_parser('play', help="Run a Sequence game")
    play_parser.add_argument('-p0',  '--player0',     dest='player0', nargs='+', default=['random'], help="Player0 class name & arguments if needed")
    play_parser.add_argument('-c1',  '--player1',     dest='player1', nargs='+', default=['random'], help="Player1 class name & arguments if needed")
    play_parser.add_argument('-c2',  '--player2',     dest='player2', nargs='+', default=None, help="Player2 class name & arguments if needed")
    play_parser.add_argument('-p3',  '--player3',     dest='player3', nargs='+', default=None, help="Player3 class name & arguments if needed")
    play_parser.add_argument('-c',   '--colors',      dest='colors',  nargs='+', default="0101", help="Players colors")
    play_parser.add_argument('-rep', '--repetitions', dest='rep',     type=int,  default=1, help="Numbers of plays to run")
    play_parser.add_argument('-ca',  '--cards',       dest='cards',   type=int,  default=6, help="Numbers of cards per player")
    play_parser.add_argument('-w',   '--win',         dest='win',     type=int,  default=2, help="Numbers of sequences needed to win")
    play_parser.add_argument('-v',   '--verbose',     dest='verbose', action='store_true', help="Print the game result at the end")
    play_parser.add_argument('-H',   '--hand',        dest='hand',    default='handout', help="Game handout strategy")
    play_parser.set_defaults(command=play)

    match_parser = subparsers.add_parser('match', help="Run a Sequence match")
    match_parser.add_argument('-p',   '--player',      dest='player',   nargs='+', default=['random'], help="Player class name & arguments if needed")
    match_parser.add_argument('-c',   '--colors',      dest='colors',   nargs='+', default="01", help="Players colors")
    match_parser.add_argument('-rep', '--repetitions', dest='rep',      type=int, default=1, help="Numbers of plays to run per oponent")
    match_parser.add_argument('-H',   '--hand',        dest='hand',     default='handout', help="Game handout strategy")
    match_parser.add_argument('-o',   '--oponents',    dest='oponents', type=str, nargs='+', required=True, help="Oponents class names")
    match_parser.add_argument('-ca',  '--cards',       dest='cards',    type=int,  default=6, help="Numbers of cards per player")
    match_parser.add_argument('-w',   '--win',         dest='win',      type=int,  default=2, help="Numbers of sequences needed to win")
    match_parser.add_argument('-v',   '--verbose',     dest='verbose',  action='store_true', help="Print the game result at the end")
    match_parser.set_defaults(command=match)

    args = parser.parse_args()

    if not hasattr(args, 'command'):
        parser.print_help()
    else:
        args.command(args)


if __name__ == '__main__':
    main()
