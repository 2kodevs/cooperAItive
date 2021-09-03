def is_end(history, pieces_per_player):
    heads = []
    player_pieces = {}
    first_move = True
    for e, *d in history:
        if e.name == 'MOVE':
            player, piece, head = d
            if first_move:
                heads = list(piece)
                first_move = False
            else:
                heads[head] = piece[piece[0] == heads[head]]
            player_pieces[player] = player_pieces.get(player, 0) + 1
    return any([pieces_per_player - x <= 2 for x in player_pieces.values()])
