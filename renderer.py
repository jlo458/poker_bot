"""
renderer.py — Pygame visual layer for the poker engine.
Completely decoupled from game logic.
"""

import pygame
import sys
from card import Card, Suit, RANK_LABELS, SUIT_SYMBOLS
from game import GameState, Street, Action, PlayerState

# ── Palette ───────────────────────────────────────────────────────────────────
FELT        = (35,  100, 60)
FELT_DARK   = (28,  80,  48)
RAIL        = (90,  55,  25)
RAIL_INNER  = (110, 70,  35)
WHITE       = (255, 255, 255)
BLACK       = (10,  10,  10)
CARD_RED    = (195, 30,  30)
CARD_BLACK  = (15,  15,  15)
CARD_BACK   = (30,  45,  130)
GOLD        = (220, 180, 50)
GOLD_DARK   = (160, 120, 20)
CHIP_YELLOW = (240, 200, 60)
GREEN_BTN   = (40,  130, 70)
RED_BTN     = (160, 40,  40)
BLUE_BTN    = (40,  80,  160)
AMBER_BTN   = (160, 110, 20)
GREY_BTN    = (80,  80,  80)
TEXT_DIM    = (180, 200, 180)
TEXT_BRIGHT = (240, 240, 240)
ACTIVE_RING = (240, 200, 50)
WINNER_RING = (80,  255, 120)
FOLDED_DIM  = (120, 140, 120)

# ── Layout ────────────────────────────────────────────────────────────────────
W, H         = 1100, 740
CARD_W       = 56
CARD_H       = 80
CARD_SMALL_W = 42
CARD_SMALL_H = 60
BTN_H        = 44
BTN_GAP      = 10
PANEL_H      = 140   # bottom action panel height

# Action panel internal layout
SLIDER_Y_OFF = 10    # y offset from panel top for slider row
BTN_Y_OFF    = 62    # y offset from panel top for buttons row


class Button:
    def __init__(self, rect, label, color, text_color=WHITE, enabled=True):
        self.rect       = pygame.Rect(rect)
        self.label      = label
        self.color      = color
        self.text_color = text_color
        self.enabled    = enabled
        self.hovered    = False

    def draw(self, surf, font):
        col = self.color if self.enabled else GREY_BTN
        if self.hovered and self.enabled:
            col = tuple(min(255, c + 35) for c in col)
        pygame.draw.rect(surf, col, self.rect, border_radius=8)
        pygame.draw.rect(surf, WHITE if self.enabled else (100, 100, 100),
                         self.rect, 2, border_radius=8)
        txt = font.render(self.label, True, self.text_color if self.enabled else (150, 150, 150))
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def hit(self, pos):
        return self.enabled and self.rect.collidepoint(pos)


class PokerRenderer:
    """
    Wraps a PokerGame and renders it with Pygame.
    Call renderer.run() to enter the event loop.
    """

    def __init__(self, game, agents: dict):
        """
        game   — a PokerGame instance
        agents — {pid: agent | None}  (None = human, handled by UI)
        """
        pygame.init()
        pygame.display.set_caption("Texas Hold'em")
        self.screen = pygame.display.set_mode((W, H))
        self.clock  = pygame.time.Clock()

        self.game      = game
        self.agents    = agents
        self.human_pid = next((pid for pid, a in agents.items() if a is None), None)

        # Fonts
        self.font_lg   = pygame.font.SysFont('Arial', 26, bold=True)
        self.font_md   = pygame.font.SysFont('Arial', 20, bold=True)
        self.font_sm   = pygame.font.SysFont('Arial', 16)
        self.font_xs   = pygame.font.SysFont('Arial', 13)
        self.font_card = pygame.font.SysFont('Arial', 22, bold=True)

        # UI state
        self.buttons: list[Button] = []
        self.raise_amount   = game.BIG_BLIND * 2
        self.raise_dragging = False
        self.slider_rect    = pygame.Rect(0, 0, 0, 0)   # zero-width = hidden
        self.message        = ""
        self.ai_delay       = 0
        self.waiting_human  = False
        self.show_showdown  = False

        # Seat positions (cx, cy) for up to 6 players
        self._seats = {
            0: (W // 2,      H - PANEL_H - 30),   # bottom  (human)
            1: (W - 170,     H // 2 - 80),         # right
            2: (W - 170,     170),                  # top-right
            3: (W // 2,      70),                   # top
            4: (170,         170),                  # top-left
            5: (170,         H // 2 - 80),          # left
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Main loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        self.game.new_hand()
        self._on_new_turn()

        while True:
            dt = self.clock.tick(60)
            self._handle_events()
            self._maybe_ai_act(dt)
            self._draw()
            pygame.display.flip()

    # ─────────────────────────────────────────────────────────────────────────
    #  Events
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_events(self):
        mouse = pygame.mouse.get_pos()
        for btn in self.buttons:
            btn.hovered = btn.hit(mouse)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()

            # Slider
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if self.slider_rect.width > 0 and self.slider_rect.collidepoint(mouse):
                    self.raise_dragging = True
            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                self.raise_dragging = False
            if ev.type == pygame.MOUSEMOTION and self.raise_dragging:
                self._update_slider(mouse[0])

            # Clicks
            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                if self.show_showdown and self.game.hand_over():
                    self._start_new_hand()
                    continue
                if self.waiting_human:
                    for btn in self.buttons:
                        if btn.hit(mouse):
                            self._human_action(btn.label)
                            break

    def _update_slider(self, mx: int):
        frac = (mx - self.slider_rect.left) / max(1, self.slider_rect.width)
        frac = max(0.0, min(1.0, frac))
        mn, mx_val = self._raise_range()
        self.raise_amount = int(mn + frac * (mx_val - mn))
        self.raise_amount = max(mn, min(self.raise_amount, mx_val))
        self._rebuild_buttons()

    def _raise_range(self):
        """Return (min_raise, max_raise) for the current human player."""
        s  = self.game.get_state()
        me = s.players[self.human_pid]
        # Min raise: must be at least one big blind more than the current bet
        min_raise = s.current_bet + self.game.BIG_BLIND
        # Max raise: all their chips (all-in), expressed as a total street bet
        max_raise = me.chips + me.bet
        min_raise = max(self.game.BIG_BLIND, min(min_raise, max_raise))
        return min_raise, max_raise

    def _human_action(self, label: str):
        if label == 'Fold':
            self.game.apply_action(Action.FOLD)
        elif label == 'Check':
            self.game.apply_action(Action.CHECK)
        elif label.startswith('Call'):
            self.game.apply_action(Action.CALL)
        elif label.startswith('Bet') or label.startswith('Raise'):
            self.game.apply_action(Action.RAISE, self.raise_amount)
        elif label == 'Deal Next Hand':
            self._start_new_hand()
            return

        self.waiting_human = False
        self.slider_rect   = pygame.Rect(0, 0, 0, 0)
        self._rebuild_buttons()
        self._check_showdown()

    def _start_new_hand(self):
        self.game.new_hand()
        self.show_showdown = False
        self.message       = ""
        self._on_new_turn()

    # ─────────────────────────────────────────────────────────────────────────
    #  AI stepping
    # ─────────────────────────────────────────────────────────────────────────

    def _maybe_ai_act(self, dt: int):
        if self.game.hand_over() or self.waiting_human or self.show_showdown:
            return
        self.ai_delay -= dt
        if self.ai_delay > 0:
            return

        if self._is_human_turn():
            self.waiting_human = True
            self._rebuild_buttons()
            return

        state = self.game.get_state()
        idx   = state.acting_player
        pid   = state.players[idx].pid
        agent = self.agents.get(pid)
        legal = self.game.legal_actions()

        if agent is not None:
            action, amount = agent.decide(state, legal)
            pname = state.players[idx].name
            extra = f" ${amount}" if action == Action.RAISE else ""
            self._show_message(f"{pname}: {action.value}{extra}")
            self.game.apply_action(action, amount)
            self.ai_delay = 650
            self._rebuild_buttons()
            self._check_showdown()

    def _on_new_turn(self):
        """Called after new_hand() — determine if human goes first."""
        self.waiting_human = self._is_human_turn()
        self._rebuild_buttons()

    def _is_human_turn(self) -> bool:
        if self.game.hand_over():
            return False
        s   = self.game.get_state()
        pid = s.players[s.acting_player].pid
        return pid == self.human_pid

    def _check_showdown(self):
        if self.game.hand_over():
            self.show_showdown = True
            state   = self.game.get_state()
            winners = state.winners or []
            results = state.hand_results or {}
            if winners:
                parts = []
                for w in winners:
                    p   = state.players[w]
                    res = results.get(w)
                    parts.append(f"{p.name}" + (f": {res.name}" if res else ""))
                self._show_message("Winner — " + " | ".join(parts) + "   (click anywhere to continue)")
            self._rebuild_buttons()

    # ─────────────────────────────────────────────────────────────────────────
    #  Button / slider building
    # ─────────────────────────────────────────────────────────────────────────

    def _rebuild_buttons(self):
        self.buttons     = []
        self.slider_rect = pygame.Rect(0, 0, 0, 0)

        if self.game.hand_over():
            # Single "Deal Next Hand" button centred in panel
            bw, bh = 200, BTN_H
            bx = W // 2 - bw // 2
            by = H - PANEL_H + (PANEL_H - bh) // 2
            self.buttons.append(Button((bx, by, bw, bh), 'Deal Next Hand', GREEN_BTN))
            return

        if not self.waiting_human:
            return

        state   = self.game.get_state()
        me      = state.players[self.human_pid]
        legal   = {a: amt for a, amt in self.game.legal_actions()}
        to_call = max(0, state.current_bet - me.bet)
        can_raise = Action.RAISE in legal

        panel_top = H - PANEL_H

        # ── Slider row (only when raise is available) ─────────────────────
        if can_raise:
            mn, mx = self._raise_range()
            self.raise_amount = max(mn, min(self.raise_amount, mx))

            slider_w  = 340
            slider_h  = 20
            slider_x  = W // 2 - slider_w // 2
            slider_y  = panel_top + SLIDER_Y_OFF + 16
            self.slider_rect = pygame.Rect(slider_x, slider_y, slider_w, slider_h)

        # ── Button row ────────────────────────────────────────────────────
        # Work out which buttons exist to centre them
        btn_specs = []  # (label, color, width)
        BW = 110

        if Action.FOLD in legal:
            btn_specs.append(('Fold', RED_BTN, BW))
        if Action.CHECK in legal:
            btn_specs.append(('Check', BLUE_BTN, BW))
        if Action.CALL in legal:
            lbl = f"Call ${to_call}"
            btn_specs.append((lbl, GREEN_BTN, BW + 20))
        if can_raise:
            mn, _ = self._raise_range()
            # "Bet" when no prior bet this street, "Raise" otherwise
            verb = "Bet" if state.current_bet == 0 else "Raise"
            lbl  = f"{verb} ${self.raise_amount}"
            btn_specs.append((lbl, AMBER_BTN, BW + 30))

        total_w = sum(w for _, _, w in btn_specs) + BTN_GAP * (len(btn_specs) - 1)
        bx      = W // 2 - total_w // 2
        by      = panel_top + BTN_Y_OFF

        for label, color, bw in btn_specs:
            self.buttons.append(Button((bx, by, bw, BTN_H), label, color))
            bx += bw + BTN_GAP

    # ─────────────────────────────────────────────────────────────────────────
    #  Drawing
    # ─────────────────────────────────────────────────────────────────────────

    def _draw(self):
        self.screen.fill(RAIL)
        self._draw_table()
        self._draw_community()
        self._draw_pot()
        self._draw_players()
        self._draw_panel()
        self._draw_slider()
        self._draw_buttons()
        self._draw_message()

    def _draw_table(self):
        pygame.draw.ellipse(self.screen, RAIL_INNER, (55, 85, W - 110, H - PANEL_H - 105))
        pygame.draw.ellipse(self.screen, FELT,       (85, 112, W - 170, H - PANEL_H - 155))

    def _draw_community(self):
        state = self.game.get_state()
        total_w = 5 * CARD_W + 4 * 8
        cx = W // 2 - total_w // 2
        cy = H // 2 - CARD_H // 2 - 30

        for i in range(5):
            rx = cx + i * (CARD_W + 8)
            if i < len(state.community):
                self._draw_card(self.screen, state.community[i], rx, cy, CARD_W, CARD_H)
            else:
                self._draw_placeholder(self.screen, rx, cy, CARD_W, CARD_H)

        labels = {
            Street.PREFLOP: 'PRE-FLOP', Street.FLOP: 'FLOP',
            Street.TURN: 'TURN', Street.RIVER: 'RIVER', Street.SHOWDOWN: 'SHOWDOWN',
        }
        t = self.font_xs.render(labels.get(state.street, ''), True, TEXT_DIM)
        self.screen.blit(t, t.get_rect(center=(W // 2, cy - 16)))

    def _draw_pot(self):
        state = self.game.get_state()
        if state.pot > 0:
            t = self.font_md.render(f'POT  ${state.pot}', True, GOLD)
            cy = H // 2 - CARD_H // 2 - 30 + CARD_H + 16
            self.screen.blit(t, t.get_rect(center=(W // 2, cy)))

    def _draw_players(self):
        state = self.game.get_state()
        # Assign visual seats: human always seat 0 (bottom)
        others = [p.pid for p in state.players if p.pid != self.human_pid]
        pid_to_seat = {self.human_pid: 0}
        for i, pid in enumerate(others):
            pid_to_seat[pid] = i + 1

        for player in state.players:
            seat = pid_to_seat.get(player.pid)
            if seat is None or seat not in self._seats:
                continue
            cx, cy    = self._seats[seat]
            is_acting = (not self.game.hand_over() and
                         state.players[state.acting_player].pid == player.pid)
            is_winner = bool(state.winners and player.pid in state.winners)
            show      = (player.pid == self.human_pid) or self.show_showdown
            self._draw_seat(player, cx, cy, is_acting, is_winner, show, state)

    def _draw_seat(self, player: PlayerState, cx, cy,
                   is_acting, is_winner, show_cards, state):
        # Glow ring
        ring = WINNER_RING if is_winner else (ACTIVE_RING if is_acting else None)
        if ring:
            pygame.draw.circle(self.screen, ring, (cx, cy), 44, 3)

        # Info box
        box_col = (45, 65, 45) if is_acting else (20, 32, 20)
        if player.folded:
            box_col = (28, 28, 28)
        box = pygame.Rect(cx - 58, cy - 30, 116, 56)
        pygame.draw.rect(self.screen, box_col, box, border_radius=8)
        pygame.draw.rect(self.screen, ring or (55, 75, 55), box, 2, border_radius=8)

        name_col = FOLDED_DIM if player.folded else TEXT_BRIGHT
        t = self.font_sm.render(player.name, True, name_col)
        self.screen.blit(t, t.get_rect(center=(cx, cy - 14)))

        t = self.font_xs.render(f'${player.chips}', True, (100, 220, 120))
        self.screen.blit(t, t.get_rect(center=(cx, cy + 3)))

        if player.bet > 0:
            t = self.font_xs.render(f'bet ${player.bet}', True, CHIP_YELLOW)
            self.screen.blit(t, t.get_rect(center=(cx, cy + 18)))

        # Dealer button
        if player.is_dealer:
            pygame.draw.circle(self.screen, GOLD,      (cx + 52, cy - 34), 12)
            pygame.draw.circle(self.screen, GOLD_DARK, (cx + 52, cy - 34), 12, 2)
            t = self.font_xs.render('D', True, BLACK)
            self.screen.blit(t, t.get_rect(center=(cx + 52, cy - 34)))

        # Cards
        cw, ch  = CARD_SMALL_W, CARD_SMALL_H
        card_y  = cy - ch - 38
        for j in range(2):
            rx = cx - cw - 3 + j * (cw + 6)
            if j < len(player.hole_cards):
                if show_cards and not player.folded:
                    self._draw_card(self.screen, player.hole_cards[j], rx, card_y, cw, ch)
                else:
                    self._draw_card_back(self.screen, rx, card_y, cw, ch)
            else:
                self._draw_placeholder(self.screen, rx, card_y, cw, ch)

        if player.folded:
            fo = pygame.Surface((cw * 2 + 6, ch), pygame.SRCALPHA)
            fo.fill((0, 0, 0, 140))
            self.screen.blit(fo, (cx - cw - 3, card_y))
            t = self.font_xs.render('FOLDED', True, (200, 90, 90))
            self.screen.blit(t, t.get_rect(center=(cx, card_y + ch // 2)))

        # Hand name at showdown
        if self.show_showdown and not player.folded:
            results = state.hand_results or {}
            if player.pid in results:
                col = WINNER_RING if (state.winners and player.pid in state.winners) else TEXT_DIM
                t   = self.font_xs.render(results[player.pid].name, True, col)
                self.screen.blit(t, t.get_rect(center=(cx, card_y - 14)))

    def _draw_panel(self):
        panel = pygame.Surface((W, PANEL_H), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 190))
        self.screen.blit(panel, (0, H - PANEL_H))
        pygame.draw.line(self.screen, RAIL_INNER, (0, H - PANEL_H), (W, H - PANEL_H), 2)

        if self.waiting_human and not self.game.hand_over():
            t = self.font_md.render("Your action", True, GOLD)
            self.screen.blit(t, (28, H - PANEL_H + 10))

    def _draw_slider(self):
        if self.slider_rect.width == 0:
            return

        mn, mx = self._raise_range()
        frac   = (self.raise_amount - mn) / max(1, mx - mn)
        frac   = max(0.0, min(1.0, frac))
        sr     = self.slider_rect

        # Track
        pygame.draw.rect(self.screen, (55, 55, 55), sr, border_radius=6)
        # Fill
        fill_w = int(frac * sr.width)
        if fill_w > 0:
            pygame.draw.rect(self.screen, GOLD_DARK,
                             (sr.left, sr.top, fill_w, sr.height), border_radius=6)
        # Thumb
        tx = sr.left + fill_w
        pygame.draw.circle(self.screen, GOLD,  (tx, sr.centery), 11)
        pygame.draw.circle(self.screen, WHITE, (tx, sr.centery), 11, 2)

        # Min / max labels
        t = self.font_xs.render(f'min ${mn}', True, TEXT_DIM)
        self.screen.blit(t, (sr.left, sr.bottom + 3))
        t = self.font_xs.render(f'all-in ${mx}', True, TEXT_DIM)
        self.screen.blit(t, t.get_rect(topright=(sr.right, sr.bottom + 3)))

        # Current amount label above thumb
        t = self.font_sm.render(f'${self.raise_amount}', True, GOLD)
        self.screen.blit(t, t.get_rect(midbottom=(tx, sr.top - 3)))

    def _draw_buttons(self):
        for btn in self.buttons:
            btn.draw(self.screen, self.font_md)

    def _draw_message(self):
        if not self.message:
            return
        t  = self.font_sm.render(self.message, True, GOLD)
        r  = t.get_rect(center=(W // 2, H - PANEL_H - 18))
        bg = pygame.Surface((r.width + 20, r.height + 8), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 150))
        self.screen.blit(bg, (r.left - 10, r.top - 4))
        self.screen.blit(t, r)

    def _show_message(self, msg: str):
        self.message = msg

    # ─────────────────────────────────────────────────────────────────────────
    #  Card drawing
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_card(self, surf, card: Card, x, y, w, h):
        pygame.draw.rect(surf, WHITE, (x, y, w, h), border_radius=5)
        pygame.draw.rect(surf, (190, 190, 190), (x, y, w, h), 1, border_radius=5)
        col  = CARD_RED if card.is_red else CARD_BLACK
        font = self.font_xs if w < 50 else self.font_card
        r    = font.render(RANK_LABELS[card.rank],  True, col)
        s    = font.render(SUIT_SYMBOLS[card.suit], True, col)
        surf.blit(r, (x + 4, y + 3))
        surf.blit(s, (x + 4, y + 3 + r.get_height()))

    def _draw_card_back(self, surf, x, y, w, h):
        pygame.draw.rect(surf, CARD_BACK, (x, y, w, h), border_radius=5)
        pygame.draw.rect(surf, (50, 70, 180), (x, y, w, h), 1, border_radius=5)
        pygame.draw.rect(surf, (50, 70, 180),
                         (x + 4, y + 4, w - 8, h - 8), 1, border_radius=3)

    def _draw_placeholder(self, surf, x, y, w, h):
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(s, (255, 255, 255, 25), (0, 0, w, h), border_radius=5)
        pygame.draw.rect(s, (255, 255, 255, 45), (0, 0, w, h), 1, border_radius=5)
        surf.blit(s, (x, y))
