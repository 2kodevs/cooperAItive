def count_min(player, piece):
    cant = [0, 0]
    for item in player.pieces:
        cant[0] += (piece[0] in item)
        cant[1] += (piece[1] in item)
    val = min(cant)
    return val, cant.index(val)

    
def game_data_collector(current_hand, player_id, history):
    pieces = [[], [], [], []]
    pieces[player_id].extend(current_hand)
    missing = [[], [], [], []]

    heads = [-1, -1]
    empty = [-1, -1]
    for event, *data in history:
        if event.name == 'MOVE':
            move, id, head = data 
            pieces[id].append(move)
            if heads == empty:
                heads = move
            else:
                heads[head] = move[move[0] == heads[head]]
        elif event.name == 'PASS':
            id = data[0]
            missing[id].extend(heads)
    return pieces, missing  
