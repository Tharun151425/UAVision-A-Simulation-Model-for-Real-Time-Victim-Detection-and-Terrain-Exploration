"""
exit_screen.py — God of War / GTA SA Style Exit Confirmation
=============================================================
Standalone Pygame exit screen with cinematic feel.

USAGE (standalone):
    python exit_screen.py

USAGE (integrate into any script):
    from exit_screen import show_exit_screen
    if show_exit_screen():
        sys.exit(0)   # user chose QUIT
    # else: user chose RESUME, continue running

RETURNS:
    True  = user wants to QUIT
    False = user wants to RESUME
"""
import sys
import math
import random
import time

try:
    import pygame
except ImportError:
    print("ERROR: pip install pygame")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
SCREEN_W, SCREEN_H = 800, 500
BG_COLOR    = (8, 8, 12)
TITLE_COLOR = (220, 180, 100)
BTN_RESUME  = (40, 160, 80)
BTN_QUIT    = (180, 40, 40)
BTN_HOVER_R = (60, 200, 100)
BTN_HOVER_Q = (220, 60, 60)
TEXT_COLOR   = (220, 220, 220)
EMBER_COLORS = [(255, 120, 20), (255, 80, 10), (200, 60, 5), (255, 160, 40)]
SMOKE_COLOR  = (30, 30, 35)


# ═══════════════════════════════════════════════════════════════
# PARTICLE SYSTEM
# ═══════════════════════════════════════════════════════════════
class Ember:
    """Floating ember particle — like God of War menu."""
    def __init__(self, w, h):
        self.x = random.uniform(0, w)
        self.y = random.uniform(h * 0.6, h * 1.2)
        self.vx = random.uniform(-0.3, 0.3)
        self.vy = random.uniform(-1.5, -0.4)
        self.life = random.uniform(0.5, 1.0)
        self.decay = random.uniform(0.003, 0.008)
        self.size = random.uniform(1.0, 3.0)
        self.color = random.choice(EMBER_COLORS)
        self.w, self.h = w, h
    
    def update(self):
        self.x += self.vx + math.sin(time.time() * 2 + self.y * 0.1) * 0.2
        self.y += self.vy
        self.life -= self.decay
        if self.life <= 0 or self.y < -10:
            self.__init__(self.w, self.h)
    
    def draw(self, surf):
        alpha = max(0, min(255, int(self.life * 255)))
        r = max(0, min(255, int(self.color[0] * self.life)))
        g = max(0, min(255, int(self.color[1] * self.life)))
        b = max(0, min(255, int(self.color[2] * self.life)))
        pygame.draw.circle(surf, (r, g, b), (int(self.x), int(self.y)), max(1, int(self.size * self.life)))


class SmokeWisp:
    """Subtle smoke wisps rising from bottom."""
    def __init__(self, w, h):
        self.x = random.uniform(0, w)
        self.y = random.uniform(h * 0.7, h * 1.1)
        self.vx = random.uniform(-0.2, 0.2)
        self.vy = random.uniform(-0.8, -0.2)
        self.life = random.uniform(0.3, 0.7)
        self.decay = random.uniform(0.002, 0.005)
        self.size = random.uniform(15, 40)
        self.w, self.h = w, h
    
    def update(self):
        self.x += self.vx + math.sin(time.time() + self.x * 0.01) * 0.3
        self.y += self.vy
        self.life -= self.decay
        if self.life <= 0 or self.y < -50:
            self.__init__(self.w, self.h)
    
    def draw(self, surf):
        alpha = max(0, min(120, int(self.life * 120)))
        s = int(self.size * (1.0 + (1.0 - self.life) * 0.5))
        smoke_surf = pygame.Surface((s * 2, s * 2), pygame.SRCALPHA)
        pygame.draw.circle(smoke_surf, (*SMOKE_COLOR, alpha), (s, s), s)
        surf.blit(smoke_surf, (int(self.x) - s, int(self.y) - s))


# ═══════════════════════════════════════════════════════════════
# BUTTON
# ═══════════════════════════════════════════════════════════════
class MenuButton:
    def __init__(self, text, x, y, w, h, color, hover_color, font):
        self.text = text
        self.rect = pygame.Rect(x - w // 2, y - h // 2, w, h)
        self.color = color
        self.hover_color = hover_color
        self.font = font
        self.hovered = False
        self._anim = 0.0
    
    def update(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)
        target = 1.0 if self.hovered else 0.0
        self._anim += (target - self._anim) * 0.15
    
    def draw(self, surf):
        # Glow behind button when hovered
        if self._anim > 0.05:
            glow_alpha = int(40 * self._anim)
            glow_surf = pygame.Surface((self.rect.w + 20, self.rect.h + 20), pygame.SRCALPHA)
            c = self.hover_color
            pygame.draw.rect(glow_surf, (*c, glow_alpha), glow_surf.get_rect(), border_radius=10)
            surf.blit(glow_surf, (self.rect.x - 10, self.rect.y - 10))
        
        # Button background
        r = int(self.color[0] + (self.hover_color[0] - self.color[0]) * self._anim)
        g = int(self.color[1] + (self.hover_color[1] - self.color[1]) * self._anim)
        b = int(self.color[2] + (self.hover_color[2] - self.color[2]) * self._anim)
        pygame.draw.rect(surf, (r, g, b), self.rect, border_radius=6)
        
        # Border
        border_c = (min(255, r + 40), min(255, g + 40), min(255, b + 40))
        pygame.draw.rect(surf, border_c, self.rect, 2, border_radius=6)
        
        # Text
        txt = self.font.render(self.text, True, TEXT_COLOR)
        tx = self.rect.centerx - txt.get_width() // 2
        ty = self.rect.centery - txt.get_height() // 2
        surf.blit(txt, (tx, ty))
    
    def clicked(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)


# ═══════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════
def show_exit_screen(existing_screen=None):
    """Show cinematic exit confirmation screen.
    
    Args:
        existing_screen: If you already have a pygame screen, pass it.
                        Otherwise a new window will be created.
    
    Returns:
        True  = user chose QUIT
        False = user chose RESUME
    """
    own_window = existing_screen is None
    
    if own_window:
        if not pygame.get_init():
            pygame.init()
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("UAVision — Exit")
    else:
        screen = existing_screen
        SCREEN_W_L = screen.get_width()
        SCREEN_H_L = screen.get_height()
    
    sw = screen.get_width()
    sh = screen.get_height()
    
    clock = pygame.time.Clock()
    
    # Fonts
    try:
        title_font = pygame.font.SysFont("georgia", 42, bold=True)
        sub_font = pygame.font.SysFont("consolas", 16)
        btn_font = pygame.font.SysFont("consolas", 20, bold=True)
    except:
        title_font = pygame.font.Font(None, 48)
        sub_font = pygame.font.Font(None, 20)
        btn_font = pygame.font.Font(None, 28)
    
    # Particles
    embers = [Ember(sw, sh) for _ in range(60)]
    smoke = [SmokeWisp(sw, sh) for _ in range(15)]
    
    # Buttons
    btn_w, btn_h = 200, 55
    gap = 40
    btn_resume = MenuButton("RESUME", sw // 2 - btn_w // 2 - gap, sh // 2 + 60,
                            btn_w, btn_h, BTN_RESUME, BTN_HOVER_R, btn_font)
    btn_quit = MenuButton("QUIT", sw // 2 + btn_w // 2 + gap, sh // 2 + 60,
                          btn_w, btn_h, BTN_QUIT, BTN_HOVER_Q, btn_font)
    
    # Fade in
    fade_alpha = 255
    start_time = time.time()
    
    while True:
        dt = clock.tick(60) / 1000.0
        mouse = pygame.mouse.get_pos()
        
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                if own_window:
                    pygame.quit()
                return True
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE or ev.key == pygame.K_q:
                    return True
                if ev.key == pygame.K_RETURN or ev.key == pygame.K_KP_ENTER:
                    return False
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if btn_resume.clicked(mouse):
                    return False
                if btn_quit.clicked(mouse):
                    if own_window:
                        pygame.quit()
                    return True
        
        # ── Update ──
        btn_resume.update(mouse)
        btn_quit.update(mouse)
        for e in embers:
            e.update()
        for s in smoke:
            s.update()
        
        # Fade in
        elapsed = time.time() - start_time
        fade_alpha = max(0, int(255 * (1.0 - elapsed / 1.0)))
        
        # ── Draw ──
        screen.fill(BG_COLOR)
        
        # Smoke wisps (behind everything)
        for s in smoke:
            s.draw(screen)
        
        # Subtle vignette gradient
        vig = pygame.Surface((sw, sh), pygame.SRCALPHA)
        for i in range(0, sh // 2, 4):
            alpha = int(80 * (1.0 - i / (sh / 2)))
            pygame.draw.rect(vig, (0, 0, 0, alpha), (0, i, sw, 4))
            pygame.draw.rect(vig, (0, 0, 0, alpha), (0, sh - i - 4, sw, 4))
        screen.blit(vig, (0, 0))
        
        # Horizontal rule (subtle gold line)
        line_y = sh // 2 - 20
        line_w = int(300 + 30 * math.sin(time.time() * 1.5))
        pygame.draw.line(screen, (120, 90, 40), 
                        (sw // 2 - line_w, line_y), (sw // 2 + line_w, line_y), 1)
        
        # Title text with subtle pulse
        pulse = 0.9 + 0.1 * math.sin(time.time() * 2)
        tc = tuple(int(c * pulse) for c in TITLE_COLOR)
        title = title_font.render("ARE YOU SURE?", True, tc)
        screen.blit(title, (sw // 2 - title.get_width() // 2, sh // 2 - 90))
        
        # Subtitle
        sub = sub_font.render("Your progress has been saved.", True, (120, 120, 130))
        screen.blit(sub, (sw // 2 - sub.get_width() // 2, sh // 2 - 30))
        
        # Buttons
        btn_resume.draw(screen)
        btn_quit.draw(screen)
        
        # Key hints
        hint_resume = sub_font.render("[ENTER]", True, (80, 80, 80))
        hint_quit = sub_font.render("[ESC]", True, (80, 80, 80))
        screen.blit(hint_resume, (btn_resume.rect.centerx - hint_resume.get_width() // 2, 
                                  btn_resume.rect.bottom + 8))
        screen.blit(hint_quit, (btn_quit.rect.centerx - hint_quit.get_width() // 2,
                                btn_quit.rect.bottom + 8))
        
        # Embers (on top)
        for e in embers:
            e.draw(screen)
        
        # Bottom text
        footer = sub_font.render("U A V I S I O N", True, (50, 50, 55))
        screen.blit(footer, (sw // 2 - footer.get_width() // 2, sh - 35))
        
        # Fade overlay
        if fade_alpha > 0:
            fade = pygame.Surface((sw, sh))
            fade.fill((0, 0, 0))
            fade.set_alpha(fade_alpha)
            screen.blit(fade, (0, 0))
        
        pygame.display.flip()
    
    return True


# ═══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pygame.init()
    result = show_exit_screen()
    print(f"\nUser chose: {'QUIT' if result else 'RESUME'}")
    pygame.quit()
