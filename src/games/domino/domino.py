import argparse
from utils import prepare_player as get_player, parse_hand
from module import get_rule, get_hand, PLAYERS, RULES, BEHAVIORS, HANDS, get_player as get_player_by_name
from module.players.player_view import PlayerView


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
        game = rule(timeout=args.time, output=args.output)
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


def real_game(args):
    players = [["alphazero", args.handouts, args.rollouts, args.path] for _ in range(4)]
    for x in args.humans:
        players[x] = ["human"]

    fixed_hands = [
        parse_hand(args.player0),
        parse_hand(args.player1),
        parse_hand(args.player2),
        parse_hand(args.player3),
    ]
    hand = lambda x,y: [PlayerView(x[:]) for x in fixed_hands]

    game = get_rule("onegame")()
    winner = game.start(*[get_player(x) for x in players], hand, *args.pieces)
    print(f"The winner is team {winner}")
    return winner

  
def get_parser():
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
    play_parser.add_argument('-n',   '--nine',        dest='pieces',  action='store_const', const=[9, 10], default=[6, 7], help="Double nine mode")
    play_parser.add_argument('-rep', '--repetitions', dest='rep',     type=int, default=1, help="Numbers of plays to run")
    play_parser.add_argument('-t',   '--timeout',     dest='time',    type=int, default=60, help="Timeout for players interaction")
    play_parser.add_argument('-H',   '--hand',        dest='hand',    default='hand_out', help="Game handout strategy")
    play_parser.add_argument('-v',   '--verbose',     dest='verbose', action='store_true', help="Print the game result at the end")
    play_parser.add_argument('-out', '--output',      dest='output',  help="File to store the game outputs", default='logs.log')

    play_parser.set_defaults(command=play)

    match_parser = subparsers.add_parser('match', help="Run a domino match")
    match_parser.add_argument('-p',    '--player',     dest='player',   nargs='+', default=['random'], help="Player class name & arguments if needed")
    match_parser.add_argument('-r',   '--rule',        dest='rule',     default='onegame', help="Game rule to use in each play")
    match_parser.add_argument('-n',   '--nine',        dest='pieces',   action='store_const', const=[9, 10], default=[6, 7], help="Double nine mode")
    match_parser.add_argument('-rep', '--repetitions', dest='rep',      type=int, default=1, help="Numbers of plays to run per oponent")
    match_parser.add_argument('-H',   '--hand',        dest='hand',     default='hand_out', help="Game handout strategy")
    match_parser.add_argument('-o',   '--oponents',    dest='oponents', type=str, nargs='+', required=True, help="Oponents class names")
    match_parser.add_argument('-t',   '--timeout',     dest='time',     type=int, default=60, help="Timeout for players interaction")
    match_parser.add_argument('-out', '--output',      dest='output',  help="File to store the game outputs", default='logs.log')
    match_parser.add_argument('-v',   '--verbose',     dest='verbose',  action='store_true', help="Print the game result at the end")
    
    match_parser.set_defaults(command=match)

    rgame_parser = subparsers.add_parser('real-game', help="Run a domino match between humans and AlphaZeros, with a real world game handout")
    rgame_parser.add_argument('-p0',   '--player0',     dest='player0',  required=True, type=str, help="Player0 pieces")
    rgame_parser.add_argument('-p1',   '--player1',     dest='player1',  required=True, type=str, help="Player1 pieces")
    rgame_parser.add_argument('-p2',   '--player2',     dest='player2',  required=True, type=str, help="Player2 pieces")
    rgame_parser.add_argument('-p3',   '--player3',     dest='player3',  required=True, type=str, help="Player3 pieces")
    rgame_parser.add_argument('-hp',   '--humans',      dest='humans',   nargs="+", required=True, type=int, help="Playes that are humans")
    rgame_parser.add_argument('-rep',  '--repetitions', dest='rep',      type=int,  default=1,     help="Numbers of plays to run per oponent")
    rgame_parser.add_argument('-H',    '--handouts',    dest='handouts', type=int,  default=10,    help="Numbers of handouts for AlphaZero")
    rgame_parser.add_argument('-r',    '--rollouts',    dest='rollouts', type=int,  default=50,    help="Numbers of rollouts for AlphaZero")
    rgame_parser.add_argument('-p',    '--path',        dest='path',     default='module/training/checkpoints', help='NN\'s full path')
    rgame_parser.add_argument('-t',    '--timeout',     dest='time',     type=int, default=60, help="Timeout for players interaction")
    rgame_parser.add_argument('-out',  '--output',      dest='output',  help="File to store the game outputs", default='logs.log')
    rgame_parser.add_argument('-n',    '--nine',        dest='pieces',   action='store_const',  const=[9, 10], default=[6, 7], help="Double nine mode")
    
    rgame_parser.set_defaults(command=real_game)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if not hasattr(args, 'command'):
        parser.print_help()
    else:
        args.command(args)


if __name__ == '__main__':
    main()
