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
from evaluator import best_hand, rank_players


# ── Monte Carlo equity estimator ──────────────────────────────────────────────

def estimate_equity(hole_cards: list[Card], community: list[Card], 
                    num_opponents: int, simulations: int = 500) -> float:
    """
    Estimate equity (win probability) via Monte Carlo simulation.
    
    Args:
        hole_cards: Your 2 hole cards [Card, Card]
        community: Community cards on board (0-5 cards)
        num_opponents: Number of active opponents
        simulations: Number of runout simulations (default 500)
    
    Returns:
        Equity in [0, 1] (win rate against all opponents)
    """
    if num_opponents == 0:
        return 1.0  # No opponents left, you win by default
    
    # Build set of used cards
    used_cards = set(hole_cards + community)
    
    # Available cards in deck
    available = [Card(rank, suit) 
                 for rank in Rank 
                 for suit in Suit 
                 if Card(rank, suit) not in used_cards]
    
    cards_needed = (5 - len(community))  # How many more community cards to deal
    
    wins = 0
    for _ in range(simulations):
        # Shuffle available cards for this simulation
        deck = available.copy()
        random.shuffle(deck)
        idx = 0
        
        # Deal opponent hole cards
        opponents_hole = {}
        for opp in range(num_opponents):
            opponents_hole[opp] = [deck[idx], deck[idx + 1]]
            idx += 2
        
        # Run out remaining community cards
        simulated_community = community + deck[idx:idx + cards_needed]
        
        # Evaluate hands
        my_hand = best_hand(hole_cards, simulated_community)
        my_result = (my_hand, 'me')
        
        # Check if we beat all opponents
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
    Equity-based agent using Monte Carlo simulation and pot odds.

    Decision logic:
      1. Calculate hand equity via Monte Carlo simulation
      2. Compare to pot odds (break-even call probability)
      3. Raise if equity > 0.75, call if equity > pot_odds, fold otherwise
    """

    def decide(self, state: GameState, legal: list) -> tuple:
        obs      = state.to_observation(self.pid)
        to_call  = obs['to_call']
        pot      = obs['pot']
        actions  = {a: amt for a, amt in legal}

        # Get our cards
        hole  = [Card.from_dict(c) for c in obs['my_hole_cards']]
        board = [Card.from_dict(c) for c in obs['community']]
        
        # Count active opponents still in hand
        num_opponents = len([p for p in obs['players'] 
                            if p['pid'] != self.pid and not p['folded']])
        
        # Calculate equity via Monte Carlo (150 simulations for speed)
        equity = estimate_equity(hole, board, num_opponents, simulations=150)
        
        # Calculate pot odds: what fraction of the new pot do we invest?
        pot_odds = to_call / (pot + to_call) if to_call > 0 else 0

        # ── Decision logic ────────────────────────────────────────────────────
        
        # Strong hand: raise if possible, otherwise call
        if equity > 0.75:
            if Action.RAISE in actions and random.random() < 0.6:
                min_r = actions[Action.RAISE]
                my_chips = self._my_chips(state)
                my_bet = state.players[self.pid].bet
                # 3x pot raise (rough)
                max_raise = min(my_chips + my_bet, min_r * 3)
                amount = random.randint(min_r, max(min_r, max_raise))
                return Action.RAISE, amount
            if Action.CALL in actions:
                return Action.CALL, actions[Action.CALL]
            return Action.CHECK, 0
        
        # Medium-strong hand: call if equity justifies it
        if equity > pot_odds:
            if to_call == 0:
                return Action.CHECK, 0
            if Action.CALL in actions:
                return Action.CALL, actions[Action.CALL]
            return Action.FOLD, 0
        
        # Weak hand: only play for free
        if Action.CHECK in actions:
            return Action.CHECK, 0
        return Action.FOLD, 0
