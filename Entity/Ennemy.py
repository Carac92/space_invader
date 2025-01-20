import arcade

from Setting import ENEMY_DROP_SPEED, SCREEN_WIDTH


class Enemy(arcade.Sprite):
    def update(self):
        self.center_x += self.change_x
        if self.left < 0 or self.right > SCREEN_WIDTH:
            self.change_x *= -1
            self.center_y -= ENEMY_DROP_SPEED
        if self.bottom < 0:
            self.remove_from_sprite_lists()
