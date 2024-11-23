import arcade
import random
import numpy as np
from PIL import Image

# Constantes
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PLAYER_SPEED = 5
BULLET_SPEED = 5
ENEMY_SPEED = 1
ENEMY_DROP_SPEED = 30
NUM_ENEMY_ROWS = 2
NUM_ENEMY_COLS = 10
BULLET_COOLDOWN = 10
ENEMY_SHOOT_PROBABILITY = 0.01
ASTEROID_LIFE = 10
AMMO_EXTRA = 10
NUM_ACTIONS = 3
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 0.1


class Bullet(arcade.Sprite):
    def update(self):
        self.center_y += self.change_y
        if self.bottom > SCREEN_HEIGHT or self.top < 0:
            self.remove_from_sprite_lists()


class Enemy(arcade.Sprite):
    def update(self):
        self.center_x += self.change_x
        if self.left < 0 or self.right > SCREEN_WIDTH:
            self.change_x *= -1
            self.center_y -= ENEMY_DROP_SPEED
        if self.bottom < 0:
            self.remove_from_sprite_lists()


class Player(arcade.Sprite):
    def __init__(self, image, scale):
        super().__init__(image, scale)
        self.cooldown = 0
        self.ammo = 0

    def shoot(self, bullet_list):
        if self.cooldown == 0 and self.ammo > 0:
            bullet = Bullet("images/bullet.png", 1)
            bullet.center_x = self.center_x
            bullet.bottom = self.top
            bullet.change_y = BULLET_SPEED
            bullet_list.append(bullet)
            self.cooldown = BULLET_COOLDOWN
            self.ammo -= 1

    def update_cooldown(self):
        if self.cooldown > 0:
            self.cooldown -= 1


class Asteroid(arcade.Sprite):
    def __init__(self, image, scale, life):
        super().__init__(image, scale)
        self.original_image = Image.open(image)
        self.current_image = self.original_image.copy()
        self.life = life

    def take_damage(self):
        self.life -= 1
        if self.life > 0:
            width, height = self.current_image.size
            self.current_image = self.current_image.crop((0, 0, width, height - 1))
            self.texture = arcade.Texture(str(self.current_image), image=self.current_image)
        else:
            self.remove_from_sprite_lists()


class SpaceInvadersGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Space Invaders - AI Controlled")
        arcade.set_background_color(arcade.color.BLACK)
        self.player = None
        self.bullet_list = None
        self.enemy_list = None
        self.enemy_bullet_list = None
        self.asteroid_list = None
        self.q_table = np.zeros((SCREEN_WIDTH, NUM_ACTIONS))
        self.score = 0
        self.reset_required = False
        self.reward = 0
        self.last_action = 0
        self.state = 0

    def setup(self):
        self.player = Player("images/player.png", 0.5)
        self.player.center_x = SCREEN_WIDTH // 2
        self.player.center_y = 50

        self.bullet_list = arcade.SpriteList()
        self.enemy_list = arcade.SpriteList()
        self.enemy_bullet_list = arcade.SpriteList()
        self.asteroid_list = arcade.SpriteList()

        for row in range(NUM_ENEMY_ROWS):
            for col in range(NUM_ENEMY_COLS):
                enemy = Enemy("images/enemy.png", 0.5)
                enemy.center_x = 80 + col * 60
                enemy.center_y = SCREEN_HEIGHT - 100 - (row * 60)
                enemy.change_x = ENEMY_SPEED
                self.enemy_list.append(enemy)

        for col in range(8):
            asteroid = Asteroid("images/asteroid.png", 1, ASTEROID_LIFE)
            asteroid.center_x = 100 + col * 90
            asteroid.center_y = SCREEN_HEIGHT // 2
            self.asteroid_list.append(asteroid)

        total_enemies = NUM_ENEMY_ROWS * NUM_ENEMY_COLS
        self.player.ammo = total_enemies + AMMO_EXTRA

        self.score = 0
        self.reset_required = False

    def game_over(self, reason):
        """
        Gère la fin de partie. Affiche un message et redémarre le jeu.
        """
        print(f"Game Over: {reason}")
        self.reset_required = True  # Marquer que le jeu doit être réinitialisé

    def detect_environment(self):
        obstacles = []
        enemies = []
        covered_by_asteroid = False

        # Détecter les missiles ennemis alignés
        for bullet in self.enemy_bullet_list:
            if abs(bullet.center_x - self.player.center_x) < 10:
                obstacles.append("missile")

        # Détecter les astéroïdes alignés
        for asteroid in self.asteroid_list:
            if abs(asteroid.center_x - self.player.center_x) < 10:
                obstacles.append("asteroid")
                if asteroid.center_y > self.player.center_y:
                    covered_by_asteroid = True

        # Détecter les ennemis alignés
        for enemy in self.enemy_list:
            if abs(enemy.center_x - self.player.center_x) < 10:
                enemies.append(enemy)

        return {
            "obstacles": obstacles,
            "enemies": enemies,
            "covered_by_asteroid": covered_by_asteroid,
        }

    def choose_action(self):
        perception = self.detect_environment()

        if "missile" in perception["obstacles"]:
            action = 0 if self.player.center_x > SCREEN_WIDTH // 2 else 1
            return action

        if "missile" in perception["obstacles"] and perception["covered_by_asteroid"]:
            return -1

        if "asteroid" in perception["obstacles"] and perception["enemies"]:
            return -1

        if perception["enemies"] and not perception["covered_by_asteroid"]:
            return 2

        return random.randint(0, NUM_ACTIONS - 1)

    def perform_action(self, action):
        if action == -1:
            return

        if action == 0:
            if self.player.center_x - PLAYER_SPEED > 0:
                self.player.center_x -= PLAYER_SPEED
            else:
                self.reward -= 5
        elif action == 1:
            if self.player.center_x + PLAYER_SPEED < SCREEN_WIDTH:
                self.player.center_x += PLAYER_SPEED
            else:
                self.reward -= 5
        elif action == 2:
            self.player.shoot(self.bullet_list)

    def on_update(self, delta_time):
        if self.reset_required:
            self.setup()
            self.reset_required = False
            return

        self.state = int(self.player.center_x)
        self.player.update_cooldown()
        self.bullet_list.update()
        self.enemy_list.update()
        self.enemy_bullet_list.update()

        self.last_action = self.choose_action()
        self.perform_action(self.last_action)

        self.enemy_shoot()

        self.reward = 0

        for bullet in self.bullet_list:
            hit_list = arcade.check_for_collision_with_list(bullet, self.enemy_list)
            if hit_list:
                bullet.remove_from_sprite_lists()
                for enemy in hit_list:
                    enemy.remove_from_sprite_lists()
                    self.score += 1
                    self.reward += 10

            asteroid_hit_list = arcade.check_for_collision_with_list(bullet, self.asteroid_list)
            if asteroid_hit_list:
                bullet.remove_from_sprite_lists()
                for asteroid in asteroid_hit_list:
                    asteroid.take_damage()
                self.reward -= 5  # Pénalité pour toucher un astéroïde

        for bullet in self.enemy_bullet_list:
            asteroid_hit_list = arcade.check_for_collision_with_list(bullet, self.asteroid_list)
            if asteroid_hit_list:
                bullet.remove_from_sprite_lists()
                for asteroid in asteroid_hit_list:
                    asteroid.take_damage()

            if arcade.check_for_collision(bullet, self.player):
                self.reward -= 100
                self.game_over("Player hit by enemy bullet")

        if self.player.ammo == 0 and len(self.enemy_list) > 0:
            self.reward -= 50
            self.game_over("Out of ammunition with enemies remaining")

        for enemy in self.enemy_list:
            if enemy.bottom < 0:
                self.reward -= 100
                self.game_over("Enemy reached the bottom of the screen")

    def on_draw(self):
        arcade.start_render()
        self.player.draw()
        self.bullet_list.draw()
        self.enemy_list.draw()
        self.enemy_bullet_list.draw()
        self.asteroid_list.draw()
        arcade.draw_text(f"Score: {self.score}", 10, 20, arcade.color.WHITE, 14)
        arcade.draw_text(f"Ammo: {self.player.ammo}", 10, 50, arcade.color.WHITE, 14)

    def enemy_shoot(self):
        for enemy in self.enemy_list:
            if random.random() < ENEMY_SHOOT_PROBABILITY:
                bullet = Bullet("images/enemy_bullet.png", 1)
                bullet.center_x = enemy.center_x
                bullet.top = enemy.bottom
                bullet.change_y = -BULLET_SPEED
                self.enemy_bullet_list.append(bullet)


if __name__ == "__main__":
    game = SpaceInvadersGame()
    game.setup()
    arcade.run()
