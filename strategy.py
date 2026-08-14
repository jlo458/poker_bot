"""
strategy.py — Shared poker strategy helpers for bot agents.

Two Phase-2 pieces live here:

1. Position awareness — classify a seat (button, blinds, UTG, …) and
   return tightness / aggression / bluff multipliers. Out of position
   plays tighter; the button plays looser and more aggressively.

2. Bluff frequency calibration — given pot and bet size, compute the
   game-theoretic bluff frequency that keeps the opponent indifferent
   to calling, then scale it for position, street, and multiway pots.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random

from game import Street


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

class Position(str, Enum):
    BUTTON = "button"
    CUTOFF = "cutoff"
    HIJACK = "hijack"
    MIDDLE = "middle"
    UTG = "utg"
    SB = "sb"
    BB = "bb"


# tightness > 1 → need a stronger hand to continue (play tighter)
# aggression > 1 → bet/raise more often and larger
# bluff_scale > 1 → bluff more often
_POSITION_MODIFIERS = {
    Position.BUTTON: {"tightness": 0.85, "aggression": 1.30, "bluff_scale": 1.25},
    Position.CUTOFF: {"tightness": 0.92, "aggression": 1.15, "bluff_scale": 1.12},
    Position.HIJACK: {"tightness": 0.97, "aggression": 1.08, "bluff_scale": 1.05},
    Position.MIDDLE: {"tightness": 1.00, "aggression": 1.00, "bluff_scale": 1.00},
    Position.UTG:    {"tightness": 1.18, "aggression": 0.85, "bluff_scale": 0.70},
    Position.SB:     {"tightness": 1.22, "aggression": 0.80, "bluff_scale": 0.65},
    Position.BB:     {"tightness": 1.08, "aggression": 0.90, "bluff_scale": 0.80},
}


@dataclass(frozen=True)
class PositionProfile:
    """How a seat should adjust strategy this decision."""
    name: Position
    in_position: bool
    tightness: float
    aggression: float
    bluff_scale: float


def classify_position(pid: int, dealer_idx: int, num_players: int) -> Position:
    """
    Named seat relative to the dealer button.

    Offset 0 is always the button. For 2-handed the dealer is also the
    small blind; we still label them BUTTON because they act last postflop.
    """
    if num_players < 2:
        return Position.MIDDLE

    offset = (pid - dealer_idx) % num_players

    if num_players == 2:
        return Position.BUTTON if offset == 0 else Position.BB

    if offset == 0:
        return Position.BUTTON
    if offset == 1:
        return Position.SB
    if offset == 2:
        return Position.BB
    if offset == 3:
        return Position.UTG
    if offset == num_players - 1:
        return Position.CUTOFF
    if offset == num_players - 2 and num_players >= 6:
        return Position.HIJACK
    return Position.MIDDLE


def action_order(dealer_idx: int, num_players: int, street: str) -> list[int]:
    """Seat order of action for a street (including players who may have folded)."""
    n = num_players
    if n == 2:
        btn, bb = dealer_idx, (dealer_idx + 1) % 2
        if street == Street.PREFLOP.name:
            return [btn, bb]
        return [bb, btn]

    sb = (dealer_idx + 1) % n
    utg = (dealer_idx + 3) % n
    if street == Street.PREFLOP.name:
        return [(utg + i) % n for i in range(n)]
    return [(sb + i) % n for i in range(n)]


def is_in_position(pid: int, dealer_idx: int, num_players: int,
                   street: str, active_pids: list[int]) -> bool:
    """True when no remaining player in the hand acts after `pid` on this street."""
    order = action_order(dealer_idx, num_players, street)
    remaining = [p for p in order if p in active_pids]
    return bool(remaining) and remaining[-1] == pid


def position_profile(pid: int, dealer_idx: int, num_players: int,
                     street: str, active_pids: list[int]) -> PositionProfile:
    """Combine named seat + whether we actually close the action."""
    name = classify_position(pid, dealer_idx, num_players)
    mods = _POSITION_MODIFIERS[name]
    ip = is_in_position(pid, dealer_idx, num_players, street, active_pids)

    tightness = mods["tightness"]
    aggression = mods["aggression"]
    bluff_scale = mods["bluff_scale"]

    # Closing the action is an extra edge even from a non-button seat
    # (e.g. BB preflop). Button/cutoff already have aggressive modifiers.
    if ip and name not in (Position.BUTTON, Position.CUTOFF):
        tightness *= 0.95
        aggression *= 1.08
        bluff_scale *= 1.08

    return PositionProfile(
        name=name,
        in_position=ip,
        tightness=tightness,
        aggression=aggression,
        bluff_scale=bluff_scale,
    )


# ---------------------------------------------------------------------------
# Bluff frequency
# ---------------------------------------------------------------------------

# Reference: a half-pot bet has α = 0.5 / 1.5 = 1/3. The documented
# 8% weak-hand bluff frequency is calibrated to that size.
_ALPHA_REF = 0.5 / 1.5  # ≈ 0.333

_STREET_BLUFF_SCALE = {
    "PREFLOP": 0.70,   # steal, not a polarised barrel
    "FLOP":    1.00,
    "TURN":    0.90,
    "RIVER":   1.10,   # polarise more; no more cards
}

_MIN_BLUFF_FREQ = 0.00
_MAX_BLUFF_FREQ = 0.35


@dataclass(frozen=True)
class BluffCalibration:
    """Breakdown of how a bluff frequency was produced (useful for tests / debug)."""
    alpha: float              # GTO share of bets that should be bluffs
    bluff_to_value: float     # bluffs per value bet = bet / pot
    probability: float        # P(bluff | this weak/air hand)
    pot: int
    bet_size: int


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def gto_alpha(pot: int, bet_size: int) -> float:
    """
    Unexploitable fraction of bets that should be bluffs.

    If you bet `bet_size` into `pot`, a calling station must be right
        pot / (pot + bet_size)
    of the time to break even. You make them indifferent by bluffing
        α = bet_size / (pot + bet_size)
    of your betting range.

    Examples:
        half-pot bet  → α = 1/3
        pot-sized bet → α = 1/2
    """
    total = pot + bet_size
    if total <= 0 or bet_size <= 0:
        return 0.0
    return bet_size / total


def bluff_to_value_ratio(pot: int, bet_size: int) -> float:
    """Bluffs per value bet (bet / pot). Half-pot → 0.5; pot-sized → 1.0."""
    if pot <= 0 or bet_size <= 0:
        return 0.0
    return bet_size / pot


class BluffFrequencyCalibrator:
    """
    Turns pot, bet size, and context into a per-hand bluff probability.

    The GTO α is the share of *bets* that should be bluffs. A heuristic
    bot decides hand-by-hand among weak holdings, so we convert α into
    P(bluff | weak) by scaling the documented base frequency with
    α / α_ref (α_ref = half-pot = 1/3). Larger bets → more bluffs;
    smaller bets → fewer. Position, street, multiway, and semi-bluff
    flags then adjust around that target.

    Caps keep the bot from turning into a maniac when α is large.
    """

    def __init__(self, base_freq: float = 0.08, rng: random.Random | None = None):
        if not 0.0 <= base_freq <= 1.0:
            raise ValueError(f"base_freq must be in [0, 1], got {base_freq}")
        self.base_freq = base_freq
        self.rng = rng or random.Random()

    def calibrate(
        self,
        pot: int,
        bet_size: int,
        *,
        profile: PositionProfile | None = None,
        street: str = "FLOP",
        num_opponents: int = 1,
        is_semi_bluff: bool = False,
        facing_bet: bool = False,
        can_steal: bool = False,
    ) -> BluffCalibration:
        alpha = gto_alpha(pot, bet_size)
        ratio = bluff_to_value_ratio(pot, bet_size)

        if alpha <= 0:
            return BluffCalibration(alpha=0.0, bluff_to_value=0.0,
                                    probability=0.0, pot=pot, bet_size=bet_size)

        # Scale the documented 8% (half-pot) to this bet size.
        freq = self.base_freq * (alpha / _ALPHA_REF)

        if profile is not None:
            freq *= profile.bluff_scale

        freq *= _STREET_BLUFF_SCALE.get(street, 1.0)

        # Multiway: fold equity collapses; bluff much less.
        n_opp = max(1, num_opponents)
        freq *= 0.70 ** (n_opp - 1)

        if is_semi_bluff:
            freq *= 1.40
        if facing_bet:
            # Bluff-raising is a smaller slice of the range.
            freq *= 0.35
        if can_steal:
            freq *= 1.50

        freq = clamp(freq, _MIN_BLUFF_FREQ, _MAX_BLUFF_FREQ)
        return BluffCalibration(
            alpha=alpha,
            bluff_to_value=ratio,
            probability=freq,
            pot=pot,
            bet_size=bet_size,
        )

    def should_bluff(self, pot: int, bet_size: int, **kwargs) -> bool:
        cal = self.calibrate(pot, bet_size, **kwargs)
        return self.rng.random() < cal.probability


def size_raise(pot: int, my_chips: int, my_bet: int, min_raise: int,
               pot_fraction: float) -> int:
    """
    Convert a pot-fraction into a legal RAISE amount (total street bet).

    `min_raise` is the engine's minimum total, `my_chips + my_bet` is the cap.
    """
    pot_fraction = max(0.0, pot_fraction)
    target = my_bet + int(round(pot * pot_fraction))
    cap = my_chips + my_bet
    amount = max(min_raise, min(target, cap))
    return amount
