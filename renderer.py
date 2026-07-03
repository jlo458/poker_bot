"""
renderer.py — Pygame visual layer for the poker engine.

Completely decoupled from game logic. Import only when running with UI.
"""

import pygame
import sys
from card import Card, Suit, RANK_LABELS, SUIT_SYMBOLS
from game import GameState, Street, Action, PlayerState

# ── Palette ──────────────────────────────────────────────────────────────────
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
GREY_BTN    = (80,  80,  80)
OVERLAY     = (0,   0,   0,  160)
TEXT_DIM    = (180, 200, 180)
TEXT_BRIGHT = (240, 240, 240)
ACTIVE_RING = (240, 200, 50)
WINNER_RING = (80,  255, 120)
FOLDED_DIM  = (120, 140, 120)

# ── Layout constants ──────────────────────────────────────────────────────────
W, H        = 1100, 720
CARD_W      = 56
CARD_H      = 80
CARD_SMALL_W= 42
CARD_SMALL_H= 60
BTN_H       = 44
BTN_GAP     = 12
PANEL_H     = 130


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
            col = tuple(min(255, c + 30) for c in col)
        pygame.draw.rect(surf, col, self.rect, border_radius=8)
        pygame.draw.rect(surf, WHITE if self.enabled else (100,100,100),
                         self.rect, 2, border_radius=8)
        alpha = 255 if self.enabled else 120
        txt = font.render(self.label, True, (*self.text_color[:3], alpha) if len(self.text_color) == 4 else self.text_color)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def hit(self, pos):
        return self.enabled and self.rect.collidepoint(pos)


class PokerRenderer:
    """
    Wraps a PokerGame and renders it via Pygame.
    Call renderer.run() to enter the event loop.
    """

    def __init__(self, game, agents: dict):
        """
        game   — a PokerGame instance
        agents — dict mapping pid -> agent (must have .decide(state, legal) method)
                 Human player should map to None (handled by UI)
        """
        pygame.init()
        pygame.display.set_caption("Texas Hold'em")
        self.screen = pygame.display.set_mode((W, H))
        self.clock  = pygame.time.Clock()

        self.game   = game
        self.agents = agents   # {pid: agent | None}
        self.human_pid = next((pid for pid, a in agents.items() if a is None), None)

        # Fonts
        self.font_lg  = pygame.font.SysFont('Arial', 26, bold=True)
        self.font_md  = pygame.font.SysFont('Arial', 20, bold=True)
        self.font_sm  = pygame.font.SysFont('Arial', 16)
        self.font_xs  = pygame.font.SysFont('Arial', 13)
        self.font_card= pygame.font.SysFont('Arial', 22, bold=True)

        # UI state
        self.buttons: list[Button] = []
        self.raise_amount   = 20
        self.raise_dragging = False
        self.slider_rect    = pygame.Rect(0, 0, 0, 0)
        self.message        = ""
        self.message_timer  = 0
        self.ai_delay       = 0      # ms until next AI move
        self.waiting_human  = False
        self.show_showdown  = False
        self.showdown_timer = 0

        # Player seat positions (x, y) for up to 6 players
        self._seat_positions = {
            0: (W//2, H - PANEL_H - 20),          # bottom (human)
            1: (W - 180, H//2 - 60),               # right
            2: (W - 180, 160),                      # top-right
            3: (W//2, 60),                          # top
            4: (180, 160),                          # top-left
            5: (180, H//2 - 60),                    # left
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Main loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        self.game.new_hand()
        self._rebuild_buttons()

        while True:
            dt = self.clock.tick(60)
            self._handle_events()
            self._maybe_ai_act(dt)
            self._draw()
            pygame.display.flip()

    # ─────────────────────────────────────────────────────────────────────────
    #  Event handling
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_events(self):
        state = self.game.get_state()
        mouse = pygame.mouse.get_pos()

        for btn in self.buttons:
            btn.hovered = btn.hit(mouse)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()

            # Slider drag
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if self.slider_rect.collidepoint(mouse):
                    self.raise_dragging = True

            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                self.raise_dragging = False

            if ev.type == pygame.MOUSEMOTION and self.raise_dragging:
                self._update_slider(mouse[0])

            # Button clicks
            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                if self.game.hand_over() and self.show_showdown:
                    self.game.new_hand()
                    self.show_showdown = False
                    self._rebuild_buttons()
                    self.waiting_human = self._is_human_turn()
                    continue

                if self.waiting_human:
                    for btn in self.buttons:
                        if btn.hit(mouse):
                            self._human_action(btn.label)

    def _update_slider(self, mx):
        rx = max(self.slider_rect.left, min(mx, self.slider_rect.right))
        frac = (rx - self.slider_rect.left) / max(1, self.slider_rect.width)
        state = self.game.get_state()
        me = state.players[self.human_pid]
        min_r = state.current_bet * 2
        max_r = me.chips + me.bet
        self.raise_amount = int(min_r + frac * (max_r - min_r))
        self.raise_amount = max(min_r, min(self.raise_amount, max_r))
        self._rebuild_buttons()

    def _human_action(self, label: str):
        legal_labels = {a.value: (a, amt) for a, amt in self.game.legal_actions()}
        if label == 'Fold':
            self.game.apply_action(Action.FOLD)
        elif label == 'Check':
            self.game.apply_action(Action.CHECK)
        elif label.startswith('Call'):
            self.game.apply_action(Action.CALL)
        elif label.startswith('Raise') or label.startswith('Bet'):
            self.game.apply_action(Action.RAISE, self.raise_amount)
        elif label == 'Deal Next Hand':
            self.game.new_hand()
            self.show_showdown = False

        self.waiting_human = False
        self._rebuild_buttons()
        self._check_showdown()

    # ─────────────────────────────────────────────────────────────────────────
    #  AI stepping
    # ─────────────────────────────────────────────────────────────────────────

    def _maybe_ai_act(self, dt):
        if self.game.hand_over() or self.waiting_human or self.show_showdown:
            return

        self.ai_delay -= dt
        if self.ai_delay > 0:
            return

        if self._is_human_turn():
            self.waiting_human = True
            self._rebuild_buttons()
            return

        state  = self.game.get_state()
        pid    = state.players[state.acting_player].pid
        agent  = self.agents.get(pid)
        legal  = self.game.legal_actions()

        if agent is not None:
            action, amount = agent.decide(state, legal)
            pname = state.players[state.acting_player].name
            self._show_message(f"{pname}: {action.value}" + (f" ${amount}" if action == Action.RAISE else ""))
            self.game.apply_action(action, amount)
            self._rebuild_buttons()
            self.ai_delay = 600   # pause between AI moves (ms)
            self._check_showdown()

    def _is_human_turn(self):
        if self.game.hand_over():
            return False
        state = self.game.get_state()
        pid = state.players[state.acting_player].pid
        return pid == self.human_pid

    def _check_showdown(self):
        if self.game.hand_over():
            self.show_showdown = True
            self.showdown_timer = 3000
            state = self.game.get_state()
            if state.winners:
                names = [state.players[w].name for w in state.winners]
                hand_results = state.hand_results or {}
                result_strs = []
                for w in state.winners:
                    res = hand_results.get(w)
                    if res:
                        result_strs.append(f"{state.players[w].name}: {res.name}")
                self._show_message("Winner: " + ", ".join(names) +
                                   (" — " + " | ".join(result_strs) if result_strs else ""))

    # ─────────────────────────────────────────────────────────────────────────
    #  Button building
    # ─────────────────────────────────────────────────────────────────────────

    def _rebuild_buttons(self):
        self.buttons = []
        if self.game.hand_over():
            bx = W // 2 - 80
            by = H - PANEL_H + 42
            self.buttons.append(Button((bx, by, 160, BTN_H), 'Deal Next Hand', GREEN_BTN))
            return

        if not self.waiting_human:
            return

        state  = self.game.get_state()
        me     = state.players[self.human_pid]
        legal  = {a: amt for a, amt in self.game.legal_actions()}
        to_call= max(0, state.current_bet - me.bet)

        bx = W // 2 - 210
        by = H - PANEL_H + 42
        bw = 100

        if Action.FOLD in legal:
            self.buttons.append(Button((bx, by, bw, BTN_H), 'Fold', RED_BTN))
        bx += bw + BTN_GAP

        if Action.CHECK in legal:
            self.buttons.append(Button((bx, by, bw, BTN_H), 'Check', BLUE_BTN))
        bx += bw + BTN_GAP

        if Action.CALL in legal:
            lbl = f'Call ${to_call}'
            self.buttons.append(Button((bx, by, bw + 20, BTN_H), lbl, GREEN_BTN))
        bx += bw + 20 + BTN_GAP

        if Action.RAISE in legal:
            min_r = state.current_bet + max(state.current_bet, self.game.BIG_BLIND)
            max_r = me.chips + me.bet
            self.raise_amount = max(min_r, min(self.raise_amount, max_r))
            lbl = f'Raise ${self.raise_amount}'
            self.buttons.append(Button((bx, by, bw + 20, BTN_H), lbl, (140, 100, 20)))
            # Slider
            sx = bx + bw + 20 + BTN_GAP
            self.slider_rect = pygame.Rect(sx, by + 10, 160, BTN_H - 20)

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
        self._draw_buttons()
        if self.slider_rect.width:
            self._draw_slider()
        self._draw_message()

    def _draw_table(self):
        # Outer rail
        pygame.draw.ellipse(self.screen, RAIL_INNER,
                            (60, 90, W - 120, H - PANEL_H - 110))
        # Felt
        pygame.draw.ellipse(self.screen, FELT,
                            (90, 115, W - 180, H - PANEL_H - 155))

    def _draw_community(self):
        state = self.game.get_state()
        cx = W // 2 - (5 * CARD_W + 4 * 8) // 2
        cy = H // 2 - CARD_H // 2 - 20

        for i in range(5):
            rx = cx + i * (CARD_W + 8)
            if i < len(state.community):
                self._draw_card(self.screen, state.community[i], rx, cy, CARD_W, CARD_H)
            else:
                self._draw_card_placeholder(self.screen, rx, cy, CARD_W, CARD_H)

        # Street label
        labels = {Street.PREFLOP:'PRE-FLOP', Street.FLOP:'FLOP',
                  Street.TURN:'TURN', Street.RIVER:'RIVER', Street.SHOWDOWN:'SHOWDOWN'}
        lbl = labels.get(state.street, '')
        t = self.font_xs.render(lbl, True, TEXT_DIM)
        self.screen.blit(t, t.get_rect(center=(W//2, cy - 16)))

    def _draw_pot(self):
        state = self.game.get_state()
        if state.pot > 0:
            txt = self.font_md.render(f'POT  ${state.pot}', True, GOLD)
            self.screen.blit(txt, txt.get_rect(center=(W//2, H//2 + CARD_H//2 + 16)))

    def _draw_players(self):
        state = self.game.get_state()
        num   = len(state.players)

        for i, player in enumerate(state.players):
            # Map seats: human always at bottom (seat 0 visually)
            if player.pid == self.human_pid:
                seat = 0
            else:
                others = [p.pid for p in state.players if p.pid != self.human_pid]
                idx = others.index(player.pid)
                seat = idx + 1

            if seat not in self._seat_positions:
                continue

            sx, sy = self._seat_positions[seat]
            is_acting = (state.acting_player == i) and not self.game.hand_over()
            is_winner = state.winners and player.pid in state.winners
            is_human  = player.pid == self.human_pid

            self._draw_player_seat(player, sx, sy, is_acting, is_winner,
                                   show_cards=(is_human or self.show_showdown))

    def _draw_player_seat(self, player: PlayerState, cx, cy,
                          is_acting, is_winner, show_cards):
        # Ring
        ring_col = WINNER_RING if is_winner else (ACTIVE_RING if is_acting else None)
        if ring_col:
            pygame.draw.circle(self.screen, ring_col, (cx, cy), 42, 3)

        # Info box
        box_col = (40, 60, 40) if is_acting else (20, 30, 20)
        if player.folded:
            box_col = (30, 30, 30)
        box = pygame.Rect(cx - 56, cy - 28, 112, 52)
        pygame.draw.rect(self.screen, box_col, box, border_radius=8)
        pygame.draw.rect(self.screen, ring_col or (60, 80, 60), box, 2, border_radius=8)

        # Name
        name_col = FOLDED_DIM if player.folded else TEXT_BRIGHT
        t = self.font_sm.render(player.name, True, name_col)
        self.screen.blit(t, t.get_rect(center=(cx, cy - 14)))

        # Chips
        t = self.font_xs.render(f'${player.chips}', True, (100, 220, 120))
        self.screen.blit(t, t.get_rect(center=(cx, cy + 2)))

        # Bet
        if player.bet > 0:
            t = self.font_xs.render(f'bet ${player.bet}', True, CHIP_YELLOW)
            self.screen.blit(t, t.get_rect(center=(cx, cy + 17)))

        # Dealer button
        if player.is_dealer:
            db = pygame.Rect(cx + 42, cy - 38, 22, 22)
            pygame.draw.circle(self.screen, GOLD, (cx + 50, cy - 38), 11)
            pygame.draw.circle(self.screen, GOLD_DARK, (cx + 50, cy - 38), 11, 2)
            t = self.font_xs.render('D', True, BLACK)
            self.screen.blit(t, t.get_rect(center=(cx + 50, cy - 38)))

        # Cards
        card_y = cy - 72
        cw, ch = CARD_SMALL_W, CARD_SMALL_H
        for j in range(2):
            rx = cx - cw - 3 + j * (cw + 6)
            if j < len(player.hole_cards):
                if show_cards and not player.folded:
                    self._draw_card(self.screen, player.hole_cards[j], rx, card_y, cw, ch)
                else:
                    self._draw_card_back(self.screen, rx, card_y, cw, ch)
            else:
                self._draw_card_placeholder(self.screen, rx, card_y, cw, ch)

        # Folded overlay
        if player.folded:
            fo = pygame.Surface((cw * 2 + 6, ch), pygame.SRCALPHA)
            fo.fill((0, 0, 0, 130))
            self.screen.blit(fo, (cx - cw - 3, card_y))
            t = self.font_xs.render('FOLDED', True, (200, 100, 100))
            self.screen.blit(t, t.get_rect(center=(cx, card_y + ch // 2)))

        # Show hand name at showdown
        if self.show_showdown and not player.folded:
            state = self.game.get_state()
            results = state.hand_results or {}
            if player.pid in results:
                hn = results[player.pid].name
                t = self.font_xs.render(hn, True, WINNER_RING if (state.winners and player.pid in state.winners) else TEXT_DIM)
                self.screen.blit(t, t.get_rect(center=(cx, card_y - 14)))

    def _draw_panel(self):
        panel = pygame.Surface((W, PANEL_H), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 180))
        self.screen.blit(panel, (0, H - PANEL_H))
        pygame.draw.line(self.screen, RAIL_INNER, (0, H - PANEL_H), (W, H - PANEL_H), 2)

        # "Your turn" label
        if self.waiting_human:
            t = self.font_md.render("Your action:", True, GOLD)
            self.screen.blit(t, (30, H - PANEL_H + 14))
        elif self.game.hand_over():
            t = self.font_md.render("Hand over — press Deal to continue", True, TEXT_DIM)
            self.screen.blit(t, t.get_rect(center=(W//2, H - PANEL_H + 18)))

    def _draw_buttons(self):
        for btn in self.buttons:
            btn.draw(self.screen, self.font_md)

    def _draw_slider(self):
        if not self.slider_rect.width:
            return
        state = self.game.get_state()
        me    = state.players[self.human_pid]
        min_r = state.current_bet + max(state.current_bet, self.game.BIG_BLIND)
        max_r = max(min_r, me.chips + me.bet)

        # Track
        pygame.draw.rect(self.screen, (60, 60, 60), self.slider_rect, border_radius=4)
        # Fill
        frac = (self.raise_amount - min_r) / max(1, max_r - min_r)
        filled = pygame.Rect(self.slider_rect.left, self.slider_rect.top,
                             int(frac * self.slider_rect.width), self.slider_rect.height)
        pygame.draw.rect(self.screen, GOLD_DARK, filled, border_radius=4)
        # Thumb
        tx = self.slider_rect.left + int(frac * self.slider_rect.width)
        pygame.draw.circle(self.screen, GOLD, (tx, self.slider_rect.centery), 10)
        pygame.draw.circle(self.screen, WHITE, (tx, self.slider_rect.centery), 10, 2)
        # Min/max labels
        t = self.font_xs.render(f'${min_r}', True, TEXT_DIM)
        self.screen.blit(t, (self.slider_rect.left, self.slider_rect.bottom + 3))
        t = self.font_xs.render(f'${max_r}', True, TEXT_DIM)
        self.screen.blit(t, t.get_rect(topright=(self.slider_rect.right, self.slider_rect.bottom + 3)))

    def _draw_message(self):
        if self.message:
            t = self.font_sm.render(self.message, True, GOLD)
            r = t.get_rect(center=(W//2, H - PANEL_H - 20))
            bg = pygame.Surface((r.width + 20, r.height + 8), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 140))
            self.screen.blit(bg, (r.left - 10, r.top - 4))
            self.screen.blit(t, r)

    def _show_message(self, msg: str):
        self.message = msg

    # ─────────────────────────────────────────────────────────────────────────
    #  Card drawing helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_card(self, surf, card: Card, x, y, w, h):
        pygame.draw.rect(surf, WHITE, (x, y, w, h), border_radius=5)
        pygame.draw.rect(surf, (180, 180, 180), (x, y, w, h), 1, border_radius=5)
        col = CARD_RED if card.is_red else CARD_BLACK
        rank = RANK_LABELS[card.rank]
        suit = SUIT_SYMBOLS[card.suit]
        font = self.font_xs if w < 50 else self.font_card
        t = font.render(rank, True, col)
        surf.blit(t, (x + 4, y + 3))
        t2 = font.render(suit, True, col)
        surf.blit(t2, (x + 4, y + 3 + t.get_height()))

    def _draw_card_back(self, surf, x, y, w, h):
        pygame.draw.rect(surf, CARD_BACK, (x, y, w, h), border_radius=5)
        pygame.draw.rect(surf, (50, 70, 170), (x, y, w, h), 1, border_radius=5)
        inner = pygame.Rect(x + 4, y + 4, w - 8, h - 8)
        pygame.draw.rect(surf, (50, 70, 180), inner, 1, border_radius=3)

    def _draw_card_placeholder(self, surf, x, y, w, h):
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        s.fill((0, 0, 0, 0))
        pygame.draw.rect(s, (255, 255, 255, 30), (0, 0, w, h), border_radius=5)
        pygame.draw.rect(s, (255, 255, 255, 50), (0, 0, w, h), 1, border_radius=5)
        surf.blit(s, (x, y))
