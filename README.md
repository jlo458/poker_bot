# PokerBot 🃏

A Texas Hold'em engine and AI training framework in Python. Built in two decoupled layers — a headless game engine for training, and an optional Pygame interface for human play.

## Project Structure

```
PokerBot/
├── main.py               # Entry point
├── game.py               # Game engine (state machine)
├── card.py               # Card & Deck primitives
├── evaluator.py          # Hand evaluation
├── renderer.py           # Pygame UI (optional)
├── train_headless.py     # Headless training script
└── agents/
    └── base_agents.py    # Built-in bots
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

The project is split into three independent layers — each only knows about the layer below it:

```
renderer.py        →  Pygame UI (optional)
game.py            →  Engine (no display dependency)
card.py / evaluator.py  →  Primitives
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

### Plugging In Your AI

In `main.py`, swap seat 0 for your agent:

```python
agents = {
    0: MyPokerAI(0),       # ← your agent
    1: SimpleRuleAgent(1),
    2: CallStationAgent(2),
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

| Agent | Behaviour |
|---|---|
| `RandomAgent` | Random legal action — noisy baseline |
| `CallStationAgent` | Always calls/checks, never raises or folds |
| `SimpleRuleAgent` | Pot odds + rough hand strength estimate |

`SimpleRuleAgent` is the intended **starting point for your AI** — its `_hand_strength()` method is the slot to replace with a Monte Carlo equity estimator or neural network.

## Rules Implemented

- Blinds: $5 / $10 (configurable in `game.py`)
- Dealer rotates left each hand, skipping broke players
- Pre-flop: action starts UTG (left of BB); BB gets a free look
- Post-flop: action starts left of the dealer
- Raises re-open action for all other active players
- Min raise = current bet + one big blind
- Max raise = player's full stack (all-in)
- Split pots on ties

## Suggested Next Steps

1. **Monte Carlo equity** — deal random cards to opponents N times, count wins → real equity %
2. **Pot odds decisions** — `if equity > to_call / (pot + to_call): call`
3. **Bet sizing** — value bet 50–100% of pot proportional to hand strength
4. **Opponent modelling** — track fold frequency and aggression per opponent
5. **Neural network** — use headless training to generate hand histories; `to_observation()` output is ready to encode as a feature vector
