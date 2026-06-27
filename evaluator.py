

from collections import Counter
from itertools import combinations
from card import Card, Rank


class HandRank:
    HIGH_CARD      = 1
    ONE_PAIR       = 2
    TWO_PAIR       = 3
    THREE_OF_A_KIND= 4
    STRAIGHT       = 5
    FLUSH          = 6
    FULL_HOUSE     = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH    = 10


HAND_NAMES = {
    HandRank.HIGH_CARD:       'High Card',
    HandRank.ONE_PAIR:        'One Pair',
    HandRank.TWO_PAIR:        'Two Pair',
    HandRank.THREE_OF_A_KIND: 'Three of a Kind',
    HandRank.STRAIGHT:        'Straight',
    HandRank.FLUSH:           'Flush',
    HandRank.FULL_HOUSE:      'Full House',
    HandRank.FOUR_OF_A_KIND:  'Four of a Kind',
    HandRank.STRAIGHT_FLUSH:  'Straight Flush',
    HandRank.ROYAL_FLUSH:     'Royal Flush',
}


class HandResult:
    """Fully comparable hand result."""
    def __init__(self, rank: int, tiebreakers: tuple):
        self.rank = rank
        self.tiebreakers = tiebreakers

    def __gt__(self, other): return (self.rank, self.tiebreakers) > (other.rank, other.tiebreakers)
    def __lt__(self, other): return (self.rank, self.tiebreakers) < (other.rank, other.tiebreakers)
    def __eq__(self, other): return (self.rank, self.tiebreakers) == (other.rank, other.tiebreakers)
    def __ge__(self, other): return not self.__lt__(other)
    def __le__(self, other): return not self.__gt__(other)

    @property
    def name(self):
        return HAND_NAMES[self.rank]

    def __repr__(self):
        return f"HandResult({self.name}, {self.tiebreakers})"


def _evaluate_five(cards: list[Card]) -> HandResult:
    ranks = sorted([c.rank for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    counts = Counter(ranks)
    freq = sorted(counts.values(), reverse=True)
    groups = sorted(counts.keys(), key=lambda r: (counts[r], r), reverse=True)

    is_flush = len(set(suits)) == 1
    unique_ranks = sorted(set(ranks))
    is_straight = (
        len(unique_ranks) == 5 and unique_ranks[-1] - unique_ranks[0] == 4
    )
    # Wheel straight: A-2-3-4-5
    is_wheel = set(ranks) == {Rank.ACE, Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE}

    if is_flush and is_straight:
        high = 5 if is_wheel else ranks[0]
        hr = HandRank.ROYAL_FLUSH if high == Rank.ACE else HandRank.STRAIGHT_FLUSH
        return HandResult(hr, (high,))

    if freq[0] == 4:
        return HandResult(HandRank.FOUR_OF_A_KIND, tuple(groups))

    if freq[:2] == [3, 2]:
        return HandResult(HandRank.FULL_HOUSE, tuple(groups))

    if is_flush:
        return HandResult(HandRank.FLUSH, tuple(ranks))

    if is_straight or is_wheel:
        high = 5 if is_wheel else ranks[0]
        return HandResult(HandRank.STRAIGHT, (high,))

    if freq[0] == 3:
        return HandResult(HandRank.THREE_OF_A_KIND, tuple(groups))

    if freq[:2] == [2, 2]:
        return HandResult(HandRank.TWO_PAIR, tuple(groups))

    if freq[0] == 2:
        return HandResult(HandRank.ONE_PAIR, tuple(groups))

    return HandResult(HandRank.HIGH_CARD, tuple(ranks))


def best_hand(hole_cards: list[Card], community: list[Card]) -> HandResult:
    """Find the best 5-card hand from up to 7 cards."""
    all_cards = hole_cards + community
    if len(all_cards) < 5:
        raise ValueError("Need at least 5 cards to evaluate")
    return max(_evaluate_five(list(combo)) for combo in combinations(all_cards, 5))


def rank_players(players_hole: dict, community: list[Card]) -> list:
    """
    Given {player_id: [card, card]} and community cards,
    return list of player_ids sorted best-to-worst hand.
    Ties are grouped together.
    """
    results = {pid: best_hand(cards, community) for pid, cards in players_hole.items()}
    sorted_ids = sorted(results, key=lambda p: results[p], reverse=True)
    return sorted_ids, results
