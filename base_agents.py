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
from strategy import (
    BluffFrequencyCalibrator,
    clamp,
    position_profile,
    size_raise,
)


# Monte Carlo equity estimator 

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


# Base class

class BaseAgent:
    def __init__(self, pid: int, rng: random.Random | None = None):
        self.pid = pid
        self.rng = rng or random.Random()

    def decide(self, state: GameState, legal: list) -> tuple:
        raise NotImplementedError

    def _to_call(self, state: GameState) -> int:
        me = state.players[self.pid]
        return max(0, state.current_bet - me.bet)

    def _my_chips(self, state: GameState) -> int:
        return state.players[self.pid].chips

    def _hand_context(self, state: GameState):
        """Shared observation + position + equity inputs for thinking bots."""
        obs = state.to_observation(self.pid)
        hole = [Card.from_dict(c) for c in obs['my_hole_cards']]
        board = [Card.from_dict(c) for c in obs['community']]
        active = [p for p in obs['players'] if not p['folded']]
        active_pids = [p['pid'] for p in active]
        num_opponents = len([p for p in active if p['pid'] != self.pid])
        profile = position_profile(
            self.pid,
            state.dealer_idx,
            len(state.players),
            obs['street'],
            active_pids,
        )
        return obs, hole, board, num_opponents, profile


# Random agent

class RandomAgent(BaseAgent):
    """Picks a uniformly random legal action. Useful baseline / opponent filler."""

    def decide(self, state: GameState, legal: list) -> tuple:
        action, min_amount = self.rng.choice(legal)
        amount = min_amount
        if action == Action.RAISE:
            me = state.players[self.pid]
            max_r = me.chips + me.bet
            amount = self.rng.randint(min_amount, max(min_amount, max_r))
        return action, amount


# Rule-based "call station"

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


# Simple rule-based agent

class SimpleRuleAgent(BaseAgent):
    """
    Equity-based agent using Monte Carlo simulation and pot odds.

    Decision logic:
      1. Calculate hand equity via Monte Carlo simulation
      2. Compare to pot odds (break-even call probability)
      3. Raise if equity clears a position-adjusted bar, call if equity
         beats pot odds, otherwise fold — or bluff at a calibrated rate
    """

    def __init__(self, pid: int, rng: random.Random | None = None,
                 equity_fn=None):
        super().__init__(pid, rng=rng)
        self._equity_fn = equity_fn or estimate_equity
        self.calibrator = BluffFrequencyCalibrator(base_freq=0.06, rng=self.rng)

    def decide(self, state: GameState, legal: list) -> tuple:
        obs, hole, board, num_opponents, profile = self._hand_context(state)
        to_call  = obs['to_call']
        pot      = obs['pot']
        actions  = {a: amt for a, amt in legal}

        equity = self._equity_fn(hole, board, num_opponents, simulations=150)
        pot_odds = to_call / (pot + to_call) if to_call > 0 else 0

        # Tighter OOP (higher bar), wider on the button.
        strong_bar = clamp(0.75 * profile.tightness, 0.58, 0.88)
        raise_freq = clamp(0.60 * profile.aggression, 0.35, 0.90)

        # Strong hand: raise if possible, otherwise call
        if equity > strong_bar:
            if Action.RAISE in actions and self.rng.random() < raise_freq:
                min_r = actions[Action.RAISE]
                my_chips = self._my_chips(state)
                my_bet = state.players[self.pid].bet
                frac = 0.75 * profile.aggression
                amount = size_raise(pot, my_chips, my_bet, min_r, frac)
                return Action.RAISE, amount
            if Action.CALL in actions:
                return Action.CALL, actions[Action.CALL]
            return Action.CHECK, 0

        # Weak / medium: maybe bluff, especially in position.
        if Action.RAISE in actions:
            min_r = actions[Action.RAISE]
            my_chips = self._my_chips(state)
            my_bet = state.players[self.pid].bet
            bluff_frac = 0.50 * profile.aggression
            bluff_size = size_raise(pot, my_chips, my_bet, min_r, bluff_frac)
            is_semi = obs['street'] in ('FLOP', 'TURN') and 0.28 <= equity < strong_bar
            can_steal = (
                obs['street'] == 'PREFLOP'
                and to_call > 0
                and to_call <= 10
                and profile.name.value in ('button', 'cutoff')
            )
            if self.calibrator.should_bluff(
                pot, bluff_size,
                profile=profile,
                street=obs['street'],
                num_opponents=num_opponents,
                is_semi_bluff=is_semi,
                facing_bet=to_call > 0 and not can_steal,
                can_steal=can_steal,
            ):
                return Action.RAISE, bluff_size

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


# Pot-odds agent (Phase 1 bot + Phase 2 position / bluff calibration)

class PotOddsAgent(BaseAgent):
    """
    Monte Carlo equity + pot odds + bet sizing, with GTO-scaled bluffs
    and positional awareness.

    Every decision:
      1. Equity — Monte Carlo win probability
      2. Pot odds — only continue when equity pays for the call
      3. Position — tighter OOP, more aggressive on the button
      4. Bet sizing — value 50–100% pot; bluffs 40–60% pot
      5. Bluff calibration — frequency tracks bet/(pot+bet) so we don't
         bluff more than is unexploitable
    """

    def __init__(
        self,
        pid: int,
        simulations: int = 400,
        bluff_freq: float = 0.08,
        value_threshold: float = 0.60,
        fold_threshold: float = 0.20,
        rng: random.Random | None = None,
        equity_fn=None,
    ):
        super().__init__(pid, rng=rng)
        self.simulations = simulations
        self.value_threshold = value_threshold
        self.fold_threshold = fold_threshold
        self._equity_fn = equity_fn or estimate_equity
        self.calibrator = BluffFrequencyCalibrator(base_freq=bluff_freq, rng=self.rng)

    def decide(self, state: GameState, legal: list) -> tuple:
        obs, hole, board, num_opponents, profile = self._hand_context(state)
        to_call = obs['to_call']
        pot = obs['pot']
        street = obs['street']
        actions = {a: amt for a, amt in legal}
        me = state.players[self.pid]

        equity = self._equity_fn(hole, board, num_opponents, simulations=self.simulations)
        pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0

        value_th = clamp(self.value_threshold * profile.tightness, 0.48, 0.78)
        fold_th = clamp(self.fold_threshold * profile.tightness, 0.12, 0.32)

        can_steal = (
            street == 'PREFLOP'
            and 0 < to_call <= 10
            and profile.name.value in ('button', 'cutoff')
        )

        # --- Value bet / raise ---
        if equity >= value_th and Action.RAISE in actions:
            # Button barrels more; OOP slow-plays / checks a bit more.
            value_bet_freq = clamp(0.72 * profile.aggression, 0.45, 0.95)
            if self.rng.random() < value_bet_freq:
                strength = (equity - value_th) / max(1e-6, 1.0 - value_th)
                pot_frac = clamp(0.50 + 0.50 * strength, 0.45, 1.10) * profile.aggression
                pot_frac = clamp(pot_frac, 0.45, 1.20)
                amount = size_raise(pot, me.chips, me.bet, actions[Action.RAISE], pot_frac)
                return Action.RAISE, amount
            if Action.CALL in actions:
                return Action.CALL, actions[Action.CALL]
            if Action.CHECK in actions:
                return Action.CHECK, 0

        # --- Bluff / semi-bluff ---
        if Action.RAISE in actions:
            is_semi = street in ('FLOP', 'TURN') and fold_th <= equity < value_th
            is_air = equity < fold_th
            # Polarised: bluff air (and semi-bluff draws). Don't turn
            # medium showdown hands into bluffs.
            want_bluff = is_air or is_semi or can_steal
            if want_bluff:
                bluff_frac = clamp(0.50 * profile.aggression, 0.40, 0.75)
                bluff_size = size_raise(
                    pot, me.chips, me.bet, actions[Action.RAISE], bluff_frac,
                )
                if self.calibrator.should_bluff(
                    pot, bluff_size,
                    profile=profile,
                    street=street,
                    num_opponents=num_opponents,
                    is_semi_bluff=is_semi,
                    facing_bet=to_call > 0 and not can_steal,
                    can_steal=can_steal,
                ):
                    return Action.RAISE, bluff_size

        # --- Facing a bet: pot odds + position-adjusted fold floor ---
        if to_call > 0:
            if equity < fold_th:
                return Action.FOLD, 0
            if equity > pot_odds:
                if Action.CALL in actions:
                    return Action.CALL, actions[Action.CALL]
            return Action.FOLD, 0

        if Action.CHECK in actions:
            return Action.CHECK, 0
        if Action.CALL in actions:
            return Action.CALL, actions[Action.CALL]
        return Action.FOLD, 0
