"""Unit tests for position awareness and bluff-frequency calibration."""

import unittest

from strategy import (
    BluffFrequencyCalibrator,
    Position,
    action_order,
    bluff_to_value_ratio,
    classify_position,
    clamp,
    gto_alpha,
    is_in_position,
    position_profile,
    size_raise,
)


class TestPosition(unittest.TestCase):
    def test_four_handed_seats(self):
        dealer = 0
        n = 4
        self.assertEqual(classify_position(0, dealer, n), Position.BUTTON)
        self.assertEqual(classify_position(1, dealer, n), Position.SB)
        self.assertEqual(classify_position(2, dealer, n), Position.BB)
        self.assertEqual(classify_position(3, dealer, n), Position.UTG)

    def test_six_handed_cutoff_and_hijack(self):
        dealer = 2
        n = 6
        self.assertEqual(classify_position(2, dealer, n), Position.BUTTON)
        self.assertEqual(classify_position(3, dealer, n), Position.SB)
        self.assertEqual(classify_position(4, dealer, n), Position.BB)
        self.assertEqual(classify_position(5, dealer, n), Position.UTG)
        self.assertEqual(classify_position(0, dealer, n), Position.HIJACK)
        self.assertEqual(classify_position(1, dealer, n), Position.CUTOFF)

    def test_heads_up_dealer_is_button(self):
        self.assertEqual(classify_position(0, 0, 2), Position.BUTTON)
        self.assertEqual(classify_position(1, 0, 2), Position.BB)

    def test_button_closes_postflop(self):
        n, dealer = 4, 0
        active = [0, 1, 2, 3]
        self.assertTrue(is_in_position(0, dealer, n, "FLOP", active))
        self.assertFalse(is_in_position(1, dealer, n, "FLOP", active))  # SB first
        self.assertTrue(is_in_position(2, dealer, n, "PREFLOP", active))  # BB last

    def test_action_order_preflop_starts_utg(self):
        self.assertEqual(action_order(0, 4, "PREFLOP"), [3, 0, 1, 2])
        self.assertEqual(action_order(0, 4, "FLOP"), [1, 2, 3, 0])

    def test_button_is_looser_and_more_aggressive_than_sb(self):
        active = [0, 1, 2, 3]
        btn = position_profile(0, 0, 4, "FLOP", active)
        sb = position_profile(1, 0, 4, "FLOP", active)
        self.assertEqual(btn.name, Position.BUTTON)
        self.assertEqual(sb.name, Position.SB)
        self.assertTrue(btn.in_position)
        self.assertFalse(sb.in_position)
        self.assertLess(btn.tightness, sb.tightness)
        self.assertGreater(btn.aggression, sb.aggression)
        self.assertGreater(btn.bluff_scale, sb.bluff_scale)

    def test_utg_tighter_than_button(self):
        active = [0, 1, 2, 3]
        btn = position_profile(0, 0, 4, "FLOP", active)
        utg = position_profile(3, 0, 4, "FLOP", active)
        self.assertGreater(utg.tightness, btn.tightness)
        self.assertLess(utg.aggression, btn.aggression)


class TestBluffCalibration(unittest.TestCase):
    def test_gto_alpha_half_pot_and_pot_sized(self):
        self.assertAlmostEqual(gto_alpha(100, 50), 1 / 3)
        self.assertAlmostEqual(gto_alpha(100, 100), 0.5)
        self.assertAlmostEqual(gto_alpha(80, 20), 0.2)
        self.assertEqual(gto_alpha(100, 0), 0.0)
        self.assertEqual(gto_alpha(0, 0), 0.0)

    def test_bluff_to_value_ratio(self):
        self.assertAlmostEqual(bluff_to_value_ratio(100, 50), 0.5)
        self.assertAlmostEqual(bluff_to_value_ratio(100, 100), 1.0)

    def test_larger_bets_bluff_more(self):
        cal = BluffFrequencyCalibrator(base_freq=0.08)
        half = cal.calibrate(100, 50, street="RIVER")
        pot = cal.calibrate(100, 100, street="RIVER")
        self.assertGreater(pot.probability, half.probability)
        self.assertAlmostEqual(half.alpha, 1 / 3)
        self.assertAlmostEqual(pot.alpha, 0.5)

    def test_button_bluffs_more_than_sb(self):
        cal = BluffFrequencyCalibrator(base_freq=0.08)
        active = [0, 1, 2, 3]
        btn = position_profile(0, 0, 4, "FLOP", active)
        sb = position_profile(1, 0, 4, "FLOP", active)
        btn_cal = cal.calibrate(100, 50, profile=btn, street="FLOP")
        sb_cal = cal.calibrate(100, 50, profile=sb, street="FLOP")
        self.assertGreater(btn_cal.probability, sb_cal.probability)

    def test_multiway_and_facing_bet_cut_frequency(self):
        cal = BluffFrequencyCalibrator(base_freq=0.08)
        hu = cal.calibrate(100, 50, street="FLOP", num_opponents=1)
        mw = cal.calibrate(100, 50, street="FLOP", num_opponents=3)
        raise_bluff = cal.calibrate(100, 50, street="FLOP", facing_bet=True)
        self.assertGreater(hu.probability, mw.probability)
        self.assertGreater(hu.probability, raise_bluff.probability)

    def test_semi_bluff_and_steal_increase_frequency(self):
        cal = BluffFrequencyCalibrator(base_freq=0.08)
        base = cal.calibrate(100, 50, street="FLOP")
        semi = cal.calibrate(100, 50, street="FLOP", is_semi_bluff=True)
        steal = cal.calibrate(80, 40, street="PREFLOP", can_steal=True)
        no_steal = cal.calibrate(80, 40, street="PREFLOP", can_steal=False)
        self.assertGreater(semi.probability, base.probability)
        self.assertGreater(steal.probability, no_steal.probability)

    def test_probability_is_capped(self):
        cal = BluffFrequencyCalibrator(base_freq=1.0)
        result = cal.calibrate(10, 1000, street="RIVER", is_semi_bluff=True, can_steal=True)
        self.assertLessEqual(result.probability, 0.35)

    def test_should_bluff_respects_rng(self):
        always = BluffFrequencyCalibrator(base_freq=0.08, rng=_ConstRng(0.0))
        never = BluffFrequencyCalibrator(base_freq=0.08, rng=_ConstRng(0.99))
        self.assertTrue(always.should_bluff(100, 50, street="FLOP"))
        self.assertFalse(never.should_bluff(100, 50, street="FLOP"))

    def test_invalid_base_freq(self):
        with self.assertRaises(ValueError):
            BluffFrequencyCalibrator(base_freq=1.5)


class TestSizingAndClamp(unittest.TestCase):
    def test_size_raise_respects_min_and_stack(self):
        self.assertEqual(size_raise(100, 1000, 0, 10, 0.5), 50)
        self.assertEqual(size_raise(100, 1000, 0, 60, 0.5), 60)
        self.assertEqual(size_raise(100, 30, 0, 10, 1.0), 30)
        self.assertEqual(size_raise(80, 500, 20, 40, 0.5), 60)

    def test_clamp(self):
        self.assertEqual(clamp(0.5, 0.2, 0.8), 0.5)
        self.assertEqual(clamp(0.1, 0.2, 0.8), 0.2)
        self.assertEqual(clamp(0.9, 0.2, 0.8), 0.8)


class _ConstRng:
    """Minimal rng: random() is constant so action branches are deterministic."""

    def __init__(self, value: float):
        self.value = value

    def random(self) -> float:
        return self.value


if __name__ == "__main__":
    unittest.main()
