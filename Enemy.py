import arcade

from Config import SCREEN_WIDTH, ENEMY_DROP_SPEED


class Enemy(arcade.Sprite):
    #Si l ennemi arrive en bas, renvoie True, Game Over
    def update(self):
        self.center_x += self.change_x
        if self.left < 0 or self.right > SCREEN_WIDTH:
            self.change_x *= -1
            self.center_y -= ENEMY_DROP_SPEED
            return False
        if self.bottom < 0:
            self.remove_from_sprite_lists()
            return True
        return False