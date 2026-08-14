"""
game.py — Texas Hold'em game engine (state machine, no rendering)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import copy

from card import Card, Deck
from evaluator import best_hand, rank_players


class Street(Enum):
    PREFLOP  = auto()
    FLOP     = auto()
    TURN     = auto()
    RIVER    = auto()
    SHOWDOWN = auto()


class Action(Enum):
    FOLD  = 'fold'
    CHECK = 'check'
    CALL  = 'call'
    RAISE = 'raise'


@dataclass
class PlayerState:
    pid: int
    name: str
    chips: int
    hole_cards: list[Card] = field(default_factory=list)
    bet: int = 0          # bet in the current street
    total_bet: int = 0    # total invested this hand
    folded: bool = False
    all_in: bool = False
    is_dealer: bool = False
    is_human: bool = False
    has_acted: bool = False   # has acted at least once this street

    def to_dict(self):
        return {
            'pid': self.pid,
            'name': self.name,
            'chips': self.chips,
            'bet': self.bet,
            'total_bet': self.total_bet,
            'folded': self.folded,
            'all_in': self.all_in,
            'is_dealer': self.is_dealer,
            'hole_cards': [c.to_dict() for c in self.hole_cards],
        }


@dataclass
class GameState:
    players: list[PlayerState]
    community: list[Card]
    pot: int
    street: Street
    current_bet: int
    acting_player: int      # index into players list
    dealer_idx: int
    hand_number: int
    last_action: Optional[tuple] = None
    winners: Optional[list] = None
    hand_results: Optional[dict] = None

    def active_players(self):
        """Players who can still act (not folded, not all-in)."""
        return [p for p in self.players if not p.folded and not p.all_in]

    def players_in_hand(self):
        """Players still in the hand (not folded)."""
        return [p for p in self.players if not p.folded]

    def to_observation(self, perspective_pid: int) -> dict:
        """
        Returns game state from one player's perspective.
        Opponent hole cards are hidden. Use this as AI input.
        """
        me = next(p for p in self.players if p.pid == perspective_pid)
        return {
            'hand_number': self.hand_number,
            'street': self.street.name,
            'pot': self.pot,
            'current_bet': self.current_bet,
            'community': [c.to_dict() for c in self.community],
            'my_hole_cards': [c.to_dict() for c in me.hole_cards],
            'my_chips': me.chips,
            'my_bet': me.bet,
            'my_total_bet': me.total_bet,
            'to_call': max(0, self.current_bet - me.bet),
            'players': [
                {
                    'pid': p.pid,
                    'name': p.name,
                    'chips': p.chips,
                    'bet': p.bet,
                    'folded': p.folded,
                    'all_in': p.all_in,
                    'is_dealer': p.is_dealer,
                    'hole_cards': [c.to_dict() for c in p.hole_cards]
                                  if p.pid == perspective_pid else None,
                }
                for p in self.players
            ],
        }


class PokerGame:
    """
    State-machine Texas Hold'em engine.

    Usage:
        game = PokerGame(num_players=4, starting_chips=1000)
        game.new_hand()
        while not game.hand_over():
            state  = game.get_state()
            legal  = game.legal_actions()
            action, amount = agent.decide(state, legal)
            game.apply_action(action, amount)
    """

    SMALL_BLIND = 5
    BIG_BLIND   = 10

    def __init__(self, num_players: int = 4, starting_chips: int = 1000,
                 human_player: int = 0):
        assert 2 <= num_players <= 9
        names = ['You', 'Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank', 'Grace', 'Hal']
        self.players = [
            PlayerState(
                pid=i,
                name=names[i],
                chips=starting_chips,
                is_human=(i == human_player),
            )
            for i in range(num_players)
        ]
        self.dealer_idx    = num_players - 1   # first new_hand() rotates to 0
        self.hand_number   = 0
        self._state: Optional[GameState] = None
        self._deck:  Optional[Deck]      = None
        self._action_history: list       = []

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def new_hand(self):
        """Reset for a new hand, rotate dealer, post blinds, deal cards."""
        self.hand_number += 1
        self._action_history = []
        self._deck = Deck()

        # Reset per-hand fields
        for p in self.players:
            p.hole_cards = []
            p.bet        = 0
            p.total_bet  = 0
            p.folded     = False
            p.all_in     = False
            p.is_dealer  = False
            p.has_acted  = False

        # Rotate dealer (skip broke players)
        self.dealer_idx = self._next_idx(self.dealer_idx, skip_broke=True)
        self.players[self.dealer_idx].is_dealer = True

        # Post blinds
        sb_idx = self._next_idx(self.dealer_idx, skip_broke=True)
        bb_idx = self._next_idx(sb_idx,          skip_broke=True)
        pot = 0
        pot += self._post_blind(sb_idx, self.SMALL_BLIND)
        pot += self._post_blind(bb_idx, self.BIG_BLIND)

        # Deal hole cards
        for p in self.players:
            if p.chips > 0 or p.all_in:
                p.hole_cards = self._deck.deal(2)

        # Pre-flop: UTG (left of BB) acts first
        first_actor = self._next_idx(bb_idx)

        self._state = GameState(
            players=self.players,
            community=[],
            pot=pot,
            street=Street.PREFLOP,
            current_bet=self.BIG_BLIND,
            acting_player=first_actor,
            dealer_idx=self.dealer_idx,
            hand_number=self.hand_number,
        )

    def get_state(self) -> GameState:
        return self._state

    def hand_over(self) -> bool:
        return self._state is None or self._state.street == Street.SHOWDOWN

    def legal_actions(self) -> list[tuple[Action, int]]:
        s      = self._state
        me     = s.players[s.acting_player]
        to_call = max(0, s.current_bet - me.bet)
        actions = [(Action.FOLD, 0)]

        if to_call == 0:
            actions.append((Action.CHECK, 0))
        else:
            actions.append((Action.CALL, min(to_call, me.chips)))

        # Can raise/bet if we have chips beyond the call
        if me.chips > to_call:
            # Min raise = current bet + one big blind increment (standard poker rule)
            # Post-flop with no bet yet: min bet = one big blind
            min_raise = s.current_bet + self.BIG_BLIND
            cap       = me.chips + me.bet   # can't commit more than we have
            actions.append((Action.RAISE, min(min_raise, cap)))

        return actions

    def apply_action(self, action: Action, amount: int = 0):
        s  = self._state
        me = s.players[s.acting_player]

        if action == Action.FOLD:
            me.folded = True

        elif action == Action.CHECK:
            pass

        elif action == Action.CALL:
            to_call = min(max(0, s.current_bet - me.bet), me.chips)
            me.chips     -= to_call
            me.bet       += to_call
            me.total_bet += to_call
            s.pot        += to_call
            if me.chips == 0:
                me.all_in = True

        elif action == Action.RAISE:
            raise_to  = min(amount, me.chips + me.bet)
            increment = raise_to - me.bet
            me.chips     -= increment
            me.total_bet += increment
            me.bet        = raise_to
            s.pot        += increment
            s.current_bet = raise_to
            if me.chips == 0:
                me.all_in = True
            # A raise re-opens action — everyone else needs to act again
            for p in s.players:
                if p.pid != me.pid and not p.folded and not p.all_in:
                    p.has_acted = False

        me.has_acted = True
        s.last_action = (me.pid, action, amount)
        self._action_history.append(copy.copy(s.last_action))

        self._advance()

    def get_action_history(self) -> list:
        return list(self._action_history)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _post_blind(self, idx: int, amount: int) -> int:
        p      = self.players[idx]
        actual = min(amount, p.chips)
        p.chips     -= actual
        p.bet        = actual
        p.total_bet  = actual
        p.has_acted  = False   # blinds don't count as "acted"
        if p.chips == 0:
            p.all_in = True
        return actual

    def _next_idx(self, from_idx: int, skip_broke: bool = False) -> int:
        """Return the next player index who can act."""
        n   = len(self.players)
        idx = (from_idx + 1) % n
        for _ in range(n):
            p = self.players[idx]
            broke  = skip_broke and p.chips == 0 and not p.all_in
            if not p.folded and not p.all_in and not broke:
                return idx
            idx = (idx + 1) % n
        return from_idx  # fallback (shouldn't happen in a valid game)

    def _betting_done(self) -> bool:
        """
        Betting is complete when every active player has:
          1. acted at least once this street, AND
          2. their current bet matches the table's current bet
             (or they are all-in)
        """
        for p in self.players:
            if p.folded or p.all_in:
                continue
            if not p.has_acted:
                return False
            if p.bet != self._state.current_bet:
                return False
        return True

    def _advance(self):
        s       = self._state
        in_hand = s.players_in_hand()

        # Everyone folded but one — instant win
        if len(in_hand) == 1:
            self._resolve([in_hand[0].pid], {})
            return

        # Betting still going — move to next player
        if not self._betting_done():
            s.acting_player = self._next_idx(s.acting_player)
            return

        # Street is over — advance 
        next_street = {
            Street.PREFLOP: Street.FLOP,
            Street.FLOP:    Street.TURN,
            Street.TURN:    Street.RIVER,
            Street.RIVER:   Street.SHOWDOWN,
        }
        s.street = next_street[s.street]

        # Reset street bets and acted flags
        for p in s.players:
            p.bet       = 0
            p.has_acted = False
        s.current_bet = 0

        # Deal community cards
        if s.street == Street.FLOP:
            s.community = self._deck.deal(3)
        elif s.street in (Street.TURN, Street.RIVER):
            s.community += self._deck.deal(1)

        # Showdown
        if s.street == Street.SHOWDOWN:
            hole_cards       = {p.pid: p.hole_cards for p in in_hand}
            sorted_pids, results = rank_players(hole_cards, s.community)
            s.hand_results   = results
            best             = results[sorted_pids[0]]
            winners          = [pid for pid, res in results.items() if res == best]
            self._resolve(winners, results)
        else:
            # Post-flop: first active player left of dealer acts first
            s.acting_player = self._next_idx(s.dealer_idx)

    def _resolve(self, winners: list[int], results: dict):
        s          = self._state
        s.street   = Street.SHOWDOWN
        s.winners  = winners
        share      = s.pot // len(winners)
        remainder  = s.pot % len(winners)
        for pid in winners:
            p = next(x for x in s.players if x.pid == pid)
            p.chips += share
        if remainder:
            p = next(x for x in s.players if x.pid == winners[0])
            p.chips += remainder
        s.pot = 0
