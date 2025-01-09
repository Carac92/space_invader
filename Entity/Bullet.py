import arcade

from Setting import SCREEN_HEIGHT


class Bullet(arcade.Sprite):
    def update(self):
        self.center_y += self.change_y
        if self.bottom > SCREEN_HEIGHT or self.top < 0:
            self.remove_from_sprite_lists()
