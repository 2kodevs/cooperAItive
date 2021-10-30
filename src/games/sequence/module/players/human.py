from .player import BasePlayer
from ..sequence import Event
from ..utils import get_board_rep, get_color, get_rep

class Human(BasePlayer):
    def __init__(self, name):
        super().__init__(f'Human::{name}')

    def log(self, data):
        e, *details = data
        if e is Event.PLAY:
            id, card, color, pos = details
            if id != self.me:
                print(f'Player number {id} with color {get_color(color)} played {get_rep(card)} at {pos}')
        elif e is Event.REMOVE:
            id, card, pos = details
            if id != self.me:
                print(f"Player number {id} used card {get_rep(card)} to remove the piece at {pos}")
        elif e is Event.SEQUENCE:
            id, color, size  = details
            print(f"Player number {id} scored a sequence of color {get_color(color)} and size {size}")
        elif e is Event.DISCARD:
            id, card = details
            if id != self.me:
                print(f"Player number {id} discarded {get_rep(card)}")
        elif e is Event.PASS:
            id = details[0]
            print(f"Player number {id} " + ["", "(a.k.a you :-P) "][id == self.me] + "pass")
        elif e is Event.WIN:
            id, color = details
            print(f"Player number {id} with color {color} wins")
        else:
            print(f"Deck refilled")
        return super().log(data)

    def filter(self, valids):
        print("Current table:\n")

        board = get_board_rep(self.board).split("\n")        
        first_line = '     '.join(x for x in "0123456789")
        board_text = '\n'.join([f'   {first_line}', *[f'{i}  {row}' for i, row in zip("0123456789", board)]])
        print(board_text)
        
        input(
            f"\nThe cards of the player {self.me} are going to be shown."
            "\nPress enter when you are ready to see them."
        )
        cards = list(self.cards)
        valids = self.valid_moves()
        print('   '.join(str(x) for x in range(self.number_of_cards)))
        print(', '.join(get_rep(card) for card in cards))

        while True:
            d = "The format to discard a card is (idx)\n" if self.can_discard else ""
            line = input(
                "\nThe format to select your move is (idx, i, j) without the parenthesis, " 
                "where idx is the index of the card you want to play, "
                "i & j are the x-axis & y-axis indexes that represent the position where you want to play.\n"
                "All the values are 0-indexed ;-)\n"
                f"{d}"
                f"Your color is {get_color(self.color)}\n"
                "Input your selection: "
            ).split(',')
            try:
                if len(line) == 1:
                    move = (cards[int(line[0])], None)
                else:
                    idx, i, j = line
                    move = (cards[int(idx)], (int(i), int(j)))
                assert move in valids
                return [move]
            except:
                print("Something went wrong. Be sure that you are using the correct format, and your selected move is a valid one.\n")
