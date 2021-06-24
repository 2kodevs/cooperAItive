def count_min(player, piece):
    cant = [0, 0]
    for item in player.pieces:
        cant[0] += (piece[0] in item)
        cant[1] += (piece[1] in item)
    val = min(cant)
    return val, cant.index(val)
    