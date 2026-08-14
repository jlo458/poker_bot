"""Decision tests for PotOddsAgent / SimpleRuleAgent position + bluffing."""

import unittest

from card import Card, Rank, Suit
from game import Action, GameState, PlayerState, PokerGame, Street
from base_agents import PotOddsAgent, SimpleRuleAgent


def _cards():
    """Enough distinct cards for 4 hole-card pairs + a flop."""
    return [
        Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES),
        Card(Rank.SEVEN, Suit.HEARTS), Card(Rank.TWO, Suit.CLUBS),
        Card(Rank.NINE, Suit.DIAMONDS), Card(Rank.FOUR, Suit.HEARTS),
        Card(Rank.THREE, Suit.SPADES), Card(Rank.EIGHT, Suit.CLUBS),
        Card(Rank.TEN, Suit.HEARTS), Card(Rank.JACK, Suit.DIAMONDS),
        Card(Rank.QUEEN, Suit.CLUBS),
    ]


def make_state(
    acting: int,
    dealer: int = 0,
    n: int = 4,
    pot: int = 100,
    current_bet: int = 0,
    street: Street = Street.FLOP,
    chips: int = 1000,
    my_bet: int = 0,
):
    deck = _cards()
    community = deck[8:11] if street != Street.PREFLOP else []
    players = []
    for i in range(n):
        hole = deck[i * 2:(i * 2) + 2]
        players.append(PlayerState(
            pid=i,
            name=f"P{i}",
            chips=chips,
            hole_cards=hole,
            bet=my_bet if i == acting else (current_bet if current_bet and i != acting else 0),
            folded=False,
            is_dealer=(i == dealer),
        ))
    # When facing a bet, opponents have matched current_bet except the actor.
    if current_bet > 0:
        for p in players:
            if p.pid != acting:
                p.bet = current_bet
            else:
                p.bet = my_bet
    return GameState(
        players=players,
        community=community,
        pot=pot,
        street=street,
        current_bet=current_bet,
        acting_player=acting,
        dealer_idx=dealer,
        hand_number=1,
    )


def flop_legal(to_call: int = 0):
    if to_call == 0:
        return [(Action.FOLD, 0), (Action.CHECK, 0), (Action.RAISE, 10)]
    return [(Action.FOLD, 0), (Action.CALL, to_call), (Action.RAISE, to_call + 10)]


class _ConstRng:
    def __init__(self, value: float):
        self.value = value

    def random(self) -> float:
        return self.value


class TestPotOddsAgent(unittest.TestCase):
    def test_strong_hand_value_bets(self):
        state = make_state(acting=0, dealer=0)  # button
        agent = PotOddsAgent(0, rng=_ConstRng(0.0), equity_fn=lambda *a, **k: 0.90)
        action, amount = agent.decide(state, flop_legal())
        self.assertEqual(action, Action.RAISE)
        self.assertGreaterEqual(amount, 10)

    def test_air_out_of_position_folds_to_a_bet(self):
        # SB facing a flop bet, equity too low, rng never bluff-raises.
        state = make_state(acting=1, dealer=0, current_bet=40, pot=80)
        agent = PotOddsAgent(1, rng=_ConstRng(0.99), equity_fn=lambda *a, **k: 0.08)
        action, _ = agent.decide(state, flop_legal(to_call=40))
        self.assertEqual(action, Action.FOLD)

    def test_air_on_button_bluffs_when_checked_to(self):
        state = make_state(acting=0, dealer=0, pot=100, current_bet=0)
        agent = PotOddsAgent(0, rng=_ConstRng(0.0), equity_fn=lambda *a, **k: 0.08)
        action, amount = agent.decide(state, flop_legal())
        self.assertEqual(action, Action.RAISE)
        self.assertGreaterEqual(amount, 10)

    def test_medium_showdown_hand_does_not_bluff_on_river(self):
        state = make_state(acting=1, dealer=0, street=Street.RIVER, pot=100)
        agent = PotOddsAgent(1, rng=_ConstRng(0.0), equity_fn=lambda *a, **k: 0.45)
        action, _ = agent.decide(state, flop_legal())
        self.assertEqual(action, Action.CHECK)

    def test_button_value_bets_thinner_than_small_blind(self):
        # 62% equity: value on the button, not strong enough OOP on the SB.
        equity = lambda *a, **k: 0.62
        btn_state = make_state(acting=0, dealer=0, street=Street.RIVER)
        sb_state = make_state(acting=1, dealer=0, street=Street.RIVER)
        btn = PotOddsAgent(0, rng=_ConstRng(0.0), equity_fn=equity)
        sb = PotOddsAgent(1, rng=_ConstRng(0.0), equity_fn=equity)
        btn_action, _ = btn.decide(btn_state, flop_legal())
        sb_action, _ = sb.decide(sb_state, flop_legal())
        self.assertEqual(btn_action, Action.RAISE)
        self.assertEqual(sb_action, Action.CHECK)

    def test_calls_when_equity_beats_pot_odds(self):
        # pot=80, to_call=20 → pot odds = 0.20; equity 0.40 should call.
        state = make_state(acting=1, dealer=0, pot=80, current_bet=20)
        agent = PotOddsAgent(1, rng=_ConstRng(0.99), equity_fn=lambda *a, **k: 0.40)
        action, amount = agent.decide(state, flop_legal(to_call=20))
        self.assertEqual(action, Action.CALL)
        self.assertEqual(amount, 20)

    def test_preflop_button_steal(self):
        state = make_state(
            acting=0, dealer=0, street=Street.PREFLOP,
            pot=15, current_bet=10, my_bet=0,
        )
        agent = PotOddsAgent(0, rng=_ConstRng(0.0), equity_fn=lambda *a, **k: 0.22)
        action, _ = agent.decide(state, flop_legal(to_call=10))
        self.assertEqual(action, Action.RAISE)


class TestSimpleRuleAgent(unittest.TestCase):
    def test_strong_hand_raises(self):
        state = make_state(acting=0, dealer=0)
        agent = SimpleRuleAgent(0, rng=_ConstRng(0.0), equity_fn=lambda *a, **k: 0.90)
        action, _ = agent.decide(state, flop_legal())
        self.assertEqual(action, Action.RAISE)

    def test_weak_hand_can_bluff(self):
        state = make_state(acting=0, dealer=0)
        agent = SimpleRuleAgent(0, rng=_ConstRng(0.0), equity_fn=lambda *a, **k: 0.10)
        action, _ = agent.decide(state, flop_legal())
        self.assertEqual(action, Action.RAISE)

    def test_weak_hand_folds_when_not_bluffing(self):
        state = make_state(acting=1, dealer=0, current_bet=30, pot=60)
        agent = SimpleRuleAgent(1, rng=_ConstRng(0.99), equity_fn=lambda *a, **k: 0.10)
        action, _ = agent.decide(state, flop_legal(to_call=30))
        self.assertEqual(action, Action.FOLD)


class TestLiveHand(unittest.TestCase):
    def test_pot_odds_agent_plays_a_full_hand(self):
        game = PokerGame(num_players=4, starting_chips=1000, human_player=-1)
        agents = {
            i: PotOddsAgent(i, simulations=40, rng=_ConstRng(0.3),
                            equity_fn=lambda *a, **k: 0.35)
            for i in range(4)
        }
        game.new_hand()
        steps = 0
        while not game.hand_over() and steps < 80:
            state = game.get_state()
            pid = state.players[state.acting_player].pid
            action, amount = agents[pid].decide(state, game.legal_actions())
            game.apply_action(action, amount)
            steps += 1
        self.assertTrue(game.hand_over())
        self.assertIsNotNone(game.get_state().winners)


if __name__ == "__main__":
    unittest.main()
