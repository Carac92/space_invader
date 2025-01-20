import arcade

class Asteroid(arcade.Sprite):
    def __init__(self, image, scale, life):
        super().__init__(image, scale)
        self.life = life

    def take_damage(self):
        self.life -= 1
        if self.life <= 0:
            self.remove_from_sprite_lists()
