"""
agents/base_agents.py — Pluggable agent interface + built-in bots.

All agents implement:
    decide(state: GameState, legal: list[tuple[Action, int]]) -> (Action, int)

The human player is represented by None in the agents dict — the renderer
handles input for them. In headless mode, pass a real agent for every seat.
"""

import random
from game import GameState, Action
from card import Card, Rank, Suit


# ── Base class ────────────────────────────────────────────────────────────────

class BaseAgent:
    def __init__(self, pid: int):
        self.pid = pid

    def decide(self, state: GameState, legal: list) -> tuple:
        raise NotImplementedError

    def _to_call(self, state: GameState) -> int:
        me = state.players[self.pid]
        return max(0, state.current_bet - me.bet)

    def _my_chips(self, state: GameState) -> int:
        return state.players[self.pid].chips


# ── Random agent ──────────────────────────────────────────────────────────────

class RandomAgent(BaseAgent):
    """Picks a uniformly random legal action. Useful baseline / opponent filler."""

    def decide(self, state: GameState, legal: list) -> tuple:
        action, min_amount = random.choice(legal)
        amount = min_amount
        if action == Action.RAISE:
            me = state.players[self.pid]
            max_r = me.chips + me.bet
            amount = random.randint(min_amount, max(min_amount, max_r))
        return action, amount


# ── Rule-based "call station" ─────────────────────────────────────────────────

class CallStationAgent(BaseAgent):
    """
    Always calls or checks, never raises, never folds.
    Simple but useful as a punching bag for testing.
    """

    def decide(self, state: GameState, legal: list) -> tuple:
        actions = {a: amt for a, amt in legal}
        if Action.CHECK in actions:
            return Action.CHECK, 0
        if Action.CALL in actions:
            return Action.CALL, actions[Action.CALL]
        return Action.FOLD, 0


# ── Simple rule-based agent ───────────────────────────────────────────────────

class SimpleRuleAgent(BaseAgent):
    """
    A basic rule-based agent using pot odds and rough hand strength.

    Decision logic:
      - Strong hand  → raise sometimes, always call
      - Medium hand  → call if pot odds are decent
      - Weak hand    → fold if facing a bet, check if free
    """

    STRONG_PAIRS = {Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK}

    def decide(self, state: GameState, legal: list) -> tuple:
        obs      = state.to_observation(self.pid)
        strength = self._hand_strength(obs)
        to_call  = obs['to_call']
        pot      = obs['pot']
        actions  = {a: amt for a, amt in legal}

        # Pot odds: if we call, what fraction of the new pot do we invest?
        pot_odds = to_call / (pot + to_call) if to_call > 0 else 0

        if strength > 0.75:
            # Strong — raise if we can, otherwise call
            if Action.RAISE in actions and random.random() < 0.6:
                min_r   = actions[Action.RAISE]
                my_chips= self._my_chips(state)
                amount  = min(min_r + random.randint(0, min_r), my_chips + state.players[self.pid].bet)
                return Action.RAISE, amount
            if Action.CALL in actions:
                return Action.CALL, actions[Action.CALL]
            return Action.CHECK, 0

        elif strength > 0.4:
            # Medium — call if pot odds justify it
            if to_call == 0:
                return Action.CHECK, 0
            if pot_odds < strength:        # odds are good enough
                return Action.CALL, actions.get(Action.CALL, 0)
            return Action.FOLD, 0

        else:
            # Weak — only play for free
            if Action.CHECK in actions:
                return Action.CHECK, 0
            return Action.FOLD, 0

    def _hand_strength(self, obs: dict) -> float:
        """
        Crude pre-flop / post-flop hand strength estimate in [0, 1].
        Replace this with a proper equity calculator or neural net later.
        """
        from card import Card, Rank, Suit, RANK_LABELS
        hole  = [Card.from_dict(c) for c in obs['my_hole_cards']]
        board = [Card.from_dict(c) for c in obs['community']]

        if not hole:
            return 0.5

        ranks  = sorted([c.rank for c in hole], reverse=True)
        suited = hole[0].suit == hole[1].suit

        if not board:
            # Pre-flop estimate (Chen-ish simplified)
            high, low = ranks[0], ranks[1]
            score = int(high) / 14.0
            if high == low:                     score += 0.25  # pair
            if suited:                          score += 0.05
            if int(high) - int(low) <= 2:       score += 0.05  # connected
            return min(score, 1.0)
        else:
            # Post-flop: count outs roughly
            all_cards = hole + board
            all_ranks  = [c.rank for c in all_cards]
            all_suits  = [c.suit for c in all_cards]
            counts     = {}
            for r in all_ranks:
                counts[r] = counts.get(r, 0) + 1
            suit_counts= {}
            for s in all_suits:
                suit_counts[s] = suit_counts.get(s, 0) + 1

            has_pair    = any(v >= 2 for v in counts.values())
            has_trips   = any(v >= 3 for v in counts.values())
            has_quads   = any(v >= 4 for v in counts.values())
            has_flush   = any(v >= 5 for v in suit_counts.values())
            has_two_pair= sum(1 for v in counts.values() if v >= 2) >= 2
            has_full    = has_trips and has_pair and not has_quads

            if has_quads:   return 0.98
            if has_full:    return 0.92
            if has_flush:   return 0.85
            if has_trips:   return 0.75
            if has_two_pair:return 0.65
            if has_pair:    return 0.50
            return 0.25
