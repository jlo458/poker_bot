# PokerBot 

A Texas Hold'em engine and AI training framework in Python. Built in two decoupled layers — a headless game engine for training, and an optional Pygame interface for human play.

## Project Structure

```
PokerBot/
├── main.py               # Entry point
├── game.py               # Game engine (state machine)
├── card.py               # Card & Deck primitives
├── evaluator.py          # Hand evaluation
├── equity.py             # Monte Carlo equity estimator
├── renderer.py           # Pygame UI (optional)
├── train_headless.py     # Headless training script
└── agents/
    └── base_agents.py    # Built-in bots (Random, CallStation, SimpleRule, PotOdds)
```

## Requirements

- Python 3.10+
- Pygame 2.x *(UI only)*

```bash
pip install pygame
```

## Usage

```bash
# Play against bots
python main.py

# Watch bots play (no human seat)
python main.py --no-human

# Headless training loop
python main.py --headless --hands 5000

# Options
python main.py --players 3 --chips 500
```

For detailed training output:
```bash
python train_headless.py --hands 2000 --verbose
```

## How It Works

The project is split into independent layers — each only knows about the layer below it:

```
renderer.py              →  Pygame UI (optional)
game.py                  →  Engine (no display dependency)
equity.py                →  Monte Carlo estimator (no game dependency)
card.py / evaluator.py   →  Primitives
```

This means you can run thousands of hands per second for training without any display.

### Game Engine (`game.py`)

The engine is a state machine. The core loop is:

```python
game = PokerGame(num_players=4, starting_chips=1000)
game.new_hand()

while not game.hand_over():
    state  = game.get_state()
    legal  = game.legal_actions()  # [(Action.FOLD, 0), (Action.CALL, 20), ...]
    game.apply_action(action, amount)
```

`legal_actions()` returns what the current player can do. The `amount` in `apply_action(Action.RAISE, amount)` is the player's **total street bet**, not the increment.

### Monte Carlo Equity (`equity.py`)

Equity is your true probability of winning the pot given your hole cards and the current board. It's calculated by simulation:

1. Take a fresh deck minus all known cards
2. Deal random hole cards to each opponent
3. Run out the remaining board randomly
4. Check if your hand wins
5. Repeat N times — `wins / N` = your equity

```python
from equity import estimate_equity

equity = estimate_equity(
    hole_cards    = my_two_cards,   # list[Card]
    community     = board_cards,    # list[Card], 0–5 cards
    num_opponents = 2,
    simulations   = 500,            # more = accurate, slower (~20ms at 500)
)
# Returns float in [0.0, 1.0]
```

Example results:
| Hand | Street | Opponents | Equity |
|---|---|---|---|
| A♠A♥ | Pre-flop | 2 | ~74% |
| 7♣2♦ | Pre-flop | 2 | ~20% |
| Top pair (K kicker) | Flop | 1 | ~67% |
| Open-ended straight draw | Flop | 1 | ~47% |

### Pot Odds

Pot odds is the break-even point for calling a bet:

```
pot_odds = to_call / (pot + to_call)

if equity > pot_odds:  → calling is profitable long-term
if equity < pot_odds:  → folding saves money long-term
```

Example: pot=$80, opponent bets $20 → pot odds = 20/100 = **20%**. If your equity is 35%, calling is correct. If it's 15%, fold.

### Plugging In Your AI

In `main.py`, swap seat 0 for your agent:

```python
agents = {
    0: MyPokerAI(0),       # ← your agent
    1: PotOddsAgent(1),
    2: SimpleRuleAgent(2),
    3: RandomAgent(3),
}
```

Your agent only needs one method:

```python
class MyAgent(BaseAgent):
    def decide(self, state: GameState, legal: list) -> tuple[Action, int]:
        obs = state.to_observation(self.pid)  # hides opponent hole cards
        # obs keys: street, pot, to_call, my_hole_cards, community, my_chips, players
        ...
        return Action.CALL, 0
```

For training, set `human_player=-1` and run headless:

```python
game = PokerGame(num_players=4, starting_chips=1000, human_player=-1)

for episode in range(100_000):
    game.new_hand()
    while not game.hand_over():
        state  = game.get_state()
        pid    = state.players[state.acting_player].pid
        action, amount = agents[pid].decide(state, game.legal_actions())
        game.apply_action(action, amount)
    # use game.get_state().winners to compute rewards
```

## Built-in Agents

| Agent | Strategy | Benchmark |
|---|---|---|
| `RandomAgent` | Random legal action | Baseline / noise |
| `CallStationAgent` | Always calls, never folds or raises | Wins pots, loses chips |
| `SimpleRuleAgent` | Heuristic hand strength estimate | Weak — no real equity |
| `PotOddsAgent` | Monte Carlo equity + pot odds + bet sizing | **Current best** |

### PotOddsAgent

The Phase 1 statistically sound bot. Every decision goes through three steps:

1. **Equity** — runs Monte Carlo simulations to get a real win probability
2. **Pot odds** — compares equity to the price of calling; only calls/raises when it's profitable
3. **Bet sizing** — value bets 50–100% of pot scaled to equity; bluffs occasionally at 40–60% pot

Tunable parameters:
```python
PotOddsAgent(
    pid           = 0,
    simulations   = 400,    # equity accuracy vs speed
    bluff_freq    = 0.08,   # 8% bluff frequency
    value_threshold = 0.60, # equity needed to bet for value
    fold_threshold  = 0.20, # equity below which we always fold to a bet
)
```

Benchmark over 600 hands vs mixed opposition:
```
PotOddsAgent   +1994 chips  (29% win rate)
SimpleRule        +6 chips  (11% win rate)
CallStation     bust        (45% win rate — wins small, loses big)
Random          bust         (3% win rate)
```

## Rules Implemented

- Blinds: $5 / $10 (configurable via `SMALL_BLIND` / `BIG_BLIND` in `game.py`)
- Dealer rotates left each hand, skipping broke players
- Pre-flop: action starts UTG (left of BB); BB gets a free look
- Post-flop: action starts left of the dealer
- Raises re-open action for all other active players
- Min raise = current bet + one big blind
- Max raise = player's full stack (all-in)
- Split pots on ties

## Next Steps

**Phase 2 — Smarter statistics**
- Opponent modelling — track each player's fold rate, aggression, and VPIP across hands; adjust strategy accordingly
- Position awareness — play tighter out of position, more aggressively on the button
- Bluff frequency calibration — bluff enough to stay unexploitable, not more

**Phase 3 — Machine learning**
- Supervised learning on real hand history datasets (public ones available online)
- Reinforcement learning via self-play — agents improve by playing each other
- CFR (Counterfactual Regret Minimisation) — the algorithm purpose-built for poker; theoretically optimal but complex to implement
