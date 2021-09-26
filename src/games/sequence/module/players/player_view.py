class PlayerView:
    def __init__(self, cards):
        self.cards = cards[:]

    def have_card(self, card):
        return card in self.cards

    def remove(self, card):
        # Only call this function if the player has such card
        self.cards.remove(card)

    def draw(self, card):
        self.cards.append(card)

    def view(self):
        def view_cards():
            for x in self.cards:
                yield x
        return view_cards
