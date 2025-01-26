import pygame

import assets
import configs
from layer import Layer


class RetryButton(pygame.sprite.Sprite):
    def __init__(self, *groups):
        self._layer = Layer.UI
        super().__init__(*groups)
        self.image = assets.get_sprite("retry")
        self.rect = self.image.get_rect(
            center=(configs.SCREEN_WIDTH / 2, configs.SCREEN_HEIGHT / 1.5))

    def update(self, game_reset_callback):
        # Check for mouse click within the button's boundaries
        mouse_pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_pos):
            if pygame.mouse.get_pressed()[0]:  # Left mouse button clicked
                game_reset_callback()
