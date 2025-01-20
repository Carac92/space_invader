import arcade

from Entity.Bullet import Bullet
from Setting import BULLET_COOLDOWN, BULLET_SPEED, SCREEN_WIDTH


class Player(arcade.Sprite):
    def __init__(self, image, scale):
        super().__init__(image, scale)
        self.cooldown = 0
        self.ammo = 0
        self.change_x = 0

    def update(self):
        self.center_x += self.change_x
        self.center_x = max(self.width / 2, min(self.center_x, SCREEN_WIDTH -self.width / 2))
        self.change_x = 0

    def shoot(self, bullet_list):
        if self.cooldown == 0 and self.ammo > 0:
            bullet = Bullet("images/bullet.png", 1)
            bullet.center_x = self.center_x
            bullet.center_y = self.center_y + BULLET_SPEED
            #bullet.bottom = self.top
            bullet.change_y = BULLET_SPEED
            bullet_list.append(bullet)
            self.cooldown = BULLET_COOLDOWN
            self.ammo -= 1

    def update_cooldown(self):
        if self.cooldown > 0:
            self.cooldown -= 1
