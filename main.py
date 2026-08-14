"""
main.py — Entry point.

Run modes:
    python main.py                  # Pygame UI, you play as seat 0
    python main.py --headless       # No UI, all bots
    python main.py --no-human       # Pygame UI, watch bots play
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='Texas Hold\'em Poker')
    parser.add_argument('--headless',  action='store_true',
                        help='Run without Pygame (for training)')
    parser.add_argument('--no-human',  action='store_true',
                        help='Pygame UI but all AI players (spectate)')
    parser.add_argument('--players',   type=int, default=4)
    parser.add_argument('--chips',     type=int, default=1000)
    parser.add_argument('--hands',     type=int, default=0,
                        help='Max hands (0 = unlimited)')
    args = parser.parse_args()

    from game import PokerGame
    from base_agents import RandomAgent, SimpleRuleAgent, CallStationAgent, PotOddsAgent

    human_seat = -1 if (args.headless or args.no_human) else 0

    game = PokerGame(
        num_players=args.players,
        starting_chips=args.chips,
        human_player=human_seat,
    )

    # Build agents — index 0 is the human seat (None = human input via UI)
    agents = {}
    for i in range(args.players):
        if i == human_seat:
            agents[i] = None   # UI handles human input
        elif i == 1:
            agents[i] = PotOddsAgent(i)
        elif i == 2:
            agents[i] = SimpleRuleAgent(i)
        elif i == 3:
            agents[i] = CallStationAgent(i)
        else:
            agents[i] = RandomAgent(i)

    # Headless
    if args.headless:
        from collections import defaultdict
        from train_headless import run_episode

        n = args.hands if args.hands > 0 else 1000
        wins = defaultdict(int)
        print(f"Headless: {n} hands, {args.players} players\n")
        for ep in range(n):
            deltas = run_episode(game, agents)
            for pid, d in deltas.items():
                if d > 0:
                    wins[pid] += 1
        print("Done. Wins:", dict(wins))
        return

    # Pygame UI
    try:
        from renderer import PokerRenderer
    except ImportError:
        print("Pygame not found. Install with:  pip install pygame")
        sys.exit(1)

    renderer = PokerRenderer(game, agents)
    renderer.run()


if __name__ == '__main__':
    main()
