"""
train_headless.py — Run poker hands without any Pygame UI.

This is your training harness. No display, no imports from renderer.py.
Run as many episodes as you like; replace agents with your AI as it develops.

Usage:
    python train_headless.py                  # 1000 hands, prints summary
    python train_headless.py --hands 50000    # longer run
    python train_headless.py --verbose        # print every action
"""

import argparse
from collections import defaultdict

from game import PokerGame, Action, Street
from base_agents import RandomAgent, SimpleRuleAgent, CallStationAgent


def run_episode(game: PokerGame, agents: dict, verbose=False) -> dict:
    """Play one hand to completion. Returns chip deltas per player."""
    chips_before = {p.pid: p.chips for p in game.players}
    game.new_hand()

    while not game.hand_over():
        state  = game.get_state()
        acting = state.players[state.acting_player]
        legal  = game.legal_actions()
        agent  = agents[acting.pid]

        action, amount = agent.decide(state, legal)

        if verbose:
            obs = state.to_observation(acting.pid)
            print(f"  [{state.street.name}] {acting.name}: {action.value}"
                  + (f" ${amount}" if action == Action.RAISE else "")
                  + f"  (pot=${state.pot}, to_call=${obs['to_call']})")

        game.apply_action(action, amount)

    state   = game.get_state()
    winners = state.winners or []
    if verbose and winners:
        results = state.hand_results or {}
        names   = [game.players[w].name for w in winners]
        hands   = [results[w].name for w in winners if w in results]
        print(f"  → Winner(s): {names}  [{', '.join(hands)}]")

    chips_after = {p.pid: p.chips for p in game.players}
    return {pid: chips_after[pid] - chips_before[pid] for pid in chips_before}


def main():
    parser = argparse.ArgumentParser(description='Headless poker training loop')
    parser.add_argument('--hands',   type=int, default=1000)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # ── Set up game ───────────────────────────────────────────────────────────
    NUM_PLAYERS = 4
    game = PokerGame(num_players=NUM_PLAYERS, starting_chips=1000,
                     human_player=-1)   # -1 = no human seat

    agents = {
        0: SimpleRuleAgent(0),    # ← swap this for your AI
        1: RandomAgent(1),
        2: CallStationAgent(2),
        3: RandomAgent(3),
    }

    # ── Run ───────────────────────────────────────────────────────────────────
    total_deltas = defaultdict(int)
    wins         = defaultdict(int)

    print(f"Running {args.hands} hands headless…\n")

    for ep in range(args.hands):
        if args.verbose:
            print(f"\n── Hand {ep + 1} ──")

        # Skip players who are bust
        active_agents = {p.pid: agents[p.pid]
                         for p in game.players if p.chips > 0}
        if len(active_agents) < 2:
            print("Not enough players with chips — stopping.")
            break

        deltas = run_episode(game, agents, verbose=args.verbose)

        for pid, delta in deltas.items():
            total_deltas[pid] += delta
            if delta > 0:
                wins[pid] += 1

        if (ep + 1) % 200 == 0 and not args.verbose:
            print(f"  Hand {ep + 1}/{args.hands} …")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 46)
    print(f"{'Player':<12} {'Agent':<20} {'Net chips':>10} {'Wins':>6}")
    print("─" * 46)
    agent_names = {
        0: 'SimpleRule',
        1: 'Random',
        2: 'CallStation',
        3: 'Random',
    }
    for p in game.players:
        print(f"{p.name:<12} {agent_names.get(p.pid, '?'):<20}"
              f" {total_deltas[p.pid]:>+10}  {wins[p.pid]:>5}")
    print("═" * 46)
    print(f"Hands played: {args.hands}")


if __name__ == '__main__':
    main()
