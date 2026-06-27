import random
from enum import IntEnum
 
 
class Suit(IntEnum):
    CLUBS    = 0
    DIAMONDS = 1
    HEARTS   = 2
    SPADES   = 3
 
 
class Rank(IntEnum):
    TWO   = 2
    THREE = 3
    FOUR  = 4
    FIVE  = 5
    SIX   = 6
    SEVEN = 7
    EIGHT = 8
    NINE  = 9
    TEN   = 10
    JACK  = 11
    QUEEN = 12
    KING  = 13
    ACE   = 14
 
 
SUIT_SYMBOLS = {
    Suit.CLUBS:    '♣',
    Suit.DIAMONDS: '♦',
    Suit.HEARTS:   '♥',
    Suit.SPADES:   '♠',
}
 
RANK_LABELS = {
    Rank.TWO:   '2', Rank.THREE: '3', Rank.FOUR:  '4', Rank.FIVE:  '5',
    Rank.SIX:   '6', Rank.SEVEN: '7', Rank.EIGHT: '8', Rank.NINE:  '9',
    Rank.TEN:   'T', Rank.JACK:  'J', Rank.QUEEN: 'Q', Rank.KING:  'K',
    Rank.ACE:   'A',
}
 
 
class Card:
    __slots__ = ('rank', 'suit')
 
    def __init__(self, rank: Rank, suit: Suit):
        self.rank = rank
        self.suit = suit
 
    def __repr__(self):
        return f"{RANK_LABELS[self.rank]}{SUIT_SYMBOLS[self.suit]}"
 
    def __eq__(self, other):
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit
 
    def __hash__(self):
        return hash((self.rank, self.suit))
 
    @property
    def is_red(self):
        return self.suit in (Suit.DIAMONDS, Suit.HEARTS)
 
    def to_dict(self):
        return {'rank': int(self.rank), 'suit': int(self.suit)}
 
    @classmethod
    def from_dict(cls, d):
        return cls(Rank(d['rank']), Suit(d['suit']))
 
 
class Deck:
    def __init__(self):
        self.cards = [Card(rank, suit) for suit in Suit for rank in Rank]
        self.shuffle()
 
    def shuffle(self):
        random.shuffle(self.cards)
 
    def deal(self, n=1) -> list[Card]:
        if len(self.cards) < n:
            raise ValueError("Not enough cards in deck")
        dealt, self.cards = self.cards[:n], self.cards[n:]
        return dealt
 
    def __len__(self):
        return len(self.cards)
