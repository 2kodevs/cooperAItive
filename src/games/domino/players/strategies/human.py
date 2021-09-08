from ..player import BasePlayer
from .utils import game_printer, hand_printer

class Human(BasePlayer):
    def __init__(self, name):
        super().__init__(f'Human::{name}')

    def log(self, data):
        e, *details = data
        if e.name == "MOVE":
            id, move, head = details
            if id != self.me:
                print(f'Player number {id} plays {move} on head {head}')
        if e.name == "PASS":
            id = details[0]
            print(f"Player number {id} " + ["", "(a.k.a you :-P) "][id == self.me] + "pass")
        return super().log(data)

    def filter(self, valids):
        print("Current table:\n")
        game_printer(self.history)
        input(f"\nThe pieces of the player {self.me} are going to be shown. \nPress enter when you are ready to see them.")
        hand_printer(self.pieces)

        while True:
            try:
                a, b, h = input("The format to select your move is (#1, #2, head) without the parenthesis, were head is 0 o 1 indicating the left and right sides of the table respectively.\nType the piece you want play: ").split(',')
                a, b, h = int(a), int(b), int(h)
                assert self.valid((a, b), h)
                for x in [(a, b), (b, a)]:
                    if x in self.pieces:
                        return [(x, h)]
            except:
                print("Something went wrong. Be sure that you are using the correct format, and your selected move is a valid one.\n")
