"""
agents/base_agents.py — Pluggable agent interface + built-in bots.

Every agent implements:
    decide(state: GameState, legal: list[tuple[Action, int]]) -> (Action, int)

Seat None in the agents dict means "human" — the renderer takes input for
that seat. In headless mode, give every seat a real agent.
"""

import random
from game import GameState, Action
from card import Card, Rank, Suit
from evaluator import best_hand


# ── Monte Carlo equity estimator ──────────────────────────────────────────────

def estimate_equity(hole_cards: list[Card], community: list[Card],
                    num_opponents: int, simulations: int = 500) -> float:
    """
    Rough win probability by dealing random runouts.

    We shuffle the remaining deck, give each opponent two hole cards, finish
    the board, and see how often our hand beats everyone. Equity comes back
    in [0, 1]. More simulations = more stable estimate, slower call.
    """
    if num_opponents == 0:
        return 1.0  # nobody left to beat

    # Cards already dealt can't show up again
    used_cards = set(hole_cards + community)
    available = [Card(rank, suit)
                 for rank in Rank
                 for suit in Suit
                 if Card(rank, suit) not in used_cards]

    cards_needed = 5 - len(community)

    wins = 0
    for _ in range(simulations):
        deck = available.copy()
        random.shuffle(deck)
        idx = 0

        opponents_hole = {}
        for opp in range(num_opponents):
            opponents_hole[opp] = [deck[idx], deck[idx + 1]]
            idx += 2

        # Finish the board with whatever's left of the shuffled deck
        simulated_community = community + deck[idx:idx + cards_needed]
        my_hand = best_hand(hole_cards, simulated_community)

        # Only count a win if we beat every opponent on this runout
        beat_all = True
        for opp in range(num_opponents):
            opp_hand = best_hand(opponents_hole[opp], simulated_community)
            if opp_hand > my_hand:
                beat_all = False
                break

        if beat_all:
            wins += 1

    return wins / simulations


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
    """Picks a uniformly random legal action. Handy as a baseline opponent."""

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
    Always checks or calls — never raises, never folds.
    Useful as a punching bag when you just want something to bet into.
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
    Plays off Monte Carlo equity vs pot odds.

    Strong equity (> ~75%) → raise a chunk of the time, otherwise call.
    Equity better than pot odds → call (or check for free).
    Otherwise → fold unless checking is free.
    """

    def decide(self, state: GameState, legal: list) -> tuple:
        obs      = state.to_observation(self.pid)
        to_call  = obs['to_call']
        pot      = obs['pot']
        actions  = {a: amt for a, amt in legal}

        hole  = [Card.from_dict(c) for c in obs['my_hole_cards']]
        board = [Card.from_dict(c) for c in obs['community']]

        num_opponents = len([p for p in obs['players']
                            if p['pid'] != self.pid and not p['folded']])

        # 150 sims keeps decisions snappy; bump this if you want tighter estimates
        equity = estimate_equity(hole, board, num_opponents, simulations=150)

        # Break-even call price: share of the pot we'd be buying
        pot_odds = to_call / (pot + to_call) if to_call > 0 else 0

        # Strong — lean toward aggression when we can
        if equity > 0.75:
            if Action.RAISE in actions and random.random() < 0.6:
                min_r = actions[Action.RAISE]
                my_chips = self._my_chips(state)
                my_bet = state.players[self.pid].bet
                # Rough sizing: up to ~3× the minimum raise
                max_raise = min(my_chips + my_bet, min_r * 3)
                amount = random.randint(min_r, max(min_r, max_raise))
                return Action.RAISE, amount
            if Action.CALL in actions:
                return Action.CALL, actions[Action.CALL]
            return Action.CHECK, 0

        # Equity covers the price of calling — take the pot odds
        if equity > pot_odds:
            if to_call == 0:
                return Action.CHECK, 0
            if Action.CALL in actions:
                return Action.CALL, actions[Action.CALL]
            return Action.FOLD, 0

        # Nothing going — only stay in if it's free
        if Action.CHECK in actions:
            return Action.CHECK, 0
        return Action.FOLD, 0
