class PlayerView:
    def __init__(self, pieces):
        self.pieces = pieces
        self.remaining = set(pieces)
        self.nums = {}
        # Doubles are counted twice
        for (x, y) in pieces:
            self.nums[x] = self.nums.get(x, 0) + 1
            self.nums[y] = self.nums.get(y, 0) + 1

    def have_num(self, num):
        return self.nums.get(num, 0) > 0

    def have_piece(self, piece):
        return piece in self.remaining

    def remove(self, piece):
        # Only call this function if the player has such piece
        self.remaining.remove(piece)
        (x, y) = piece
        self.nums[x] -= 1
        self.nums[y] -= 1

    def total(self):
        return len(self.remaining)

    def have_move(self, piece):
        (x, y) = piece
        return self.have_num(x) or self.have_num(y)

    def points(self):
        return sum(sum(piece) for piece in self.remaining)
        