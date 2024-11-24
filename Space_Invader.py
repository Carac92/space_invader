import arcade
import random
import numpy as np
import os
import pickle

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
ENEMY_SHOOT_PROBABILITY = 0.0075
ASTEROID_LIFE = 5
AMMO_EXTRA = 10000000
NUM_ACTIONS = 4  # Ajout de l'action "Ne Rien Faire"
LEARNING_RATE = 0.05  # Taux d'apprentissage ajusté
DISCOUNT_FACTOR = 0.98  # Facteur de décote ajusté
EPSILON = 1.0  # Commence avec une exploration maximale
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999  # Décroissance plus lente de l'exploration
NUM_BINS = 10  # Réduction du nombre de bins pour gérer la taille de l'espace d'état

# Portées des zones de détection
ASTEROID_DETECTION_RANGE = 100
ENEMY_DETECTION_RANGE = 200
ENEMY_BULLET_DETECTION_RANGE = 150

# Récompenses
HIT_ASTEROID_REWARD = -15
HIT_NOTHING_REWARD = -10
ACTION_REWARD= -1
HIT_ENEMIES_REWARD = 100
LOOSE_REWARD = -10000
WIN_REWARD = 5000
DODGE_REWARD = 0
POSITION_REWARD = 0
DODGE_ASTEROID_REWARD = 0



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
        self.change_x = 0

    def update(self):
        self.center_x += self.change_x
        self.center_x = max(self.width / 2, min(SCREEN_WIDTH - self.width / 2, self.center_x))
        self.change_x = 0  # Réinitialise le mouvement après l'action

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
        self.life = life

    def take_damage(self):
        self.life -= 1
        if self.life <= 0:
            self.remove_from_sprite_lists()


class SpaceInvadersGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Space Invaders - IA avec Pénalités Ajustées")
        arcade.set_background_color(arcade.color.BLACK)
        self.player = None
        self.bullet_list = None
        self.enemy_list = None
        self.enemy_bullet_list = None
        self.asteroid_list = None
        self.epsilon = EPSILON
        self.episode = 0
        self.score = 0
        self.reset_required = False
        self.reward = 0
        self.last_action = 0
        self.state = None

        # Utiliser un dictionnaire pour la table Q
        self.q_table = {}
        self.load_q_table()

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
        print(f"Game Over: {reason}")
        self.reset_required = True
        self.episode += 1
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self.save_q_table()

    def discretize(self, value, bins=NUM_BINS, range_min=0, range_max=SCREEN_WIDTH):
        bin_size = (range_max - range_min) / bins
        index = int((value - range_min) // bin_size)
        return min(max(index, 0), bins - 1)

    def get_relative_enemy_position(self):
        if self.enemy_list:
            nearest_enemy = min(self.enemy_list, key=lambda e: abs(e.center_x - self.player.center_x))
            relative_x = nearest_enemy.center_x - self.player.center_x + SCREEN_WIDTH / 2
            relative_y = nearest_enemy.center_y - self.player.center_y
            return (self.discretize(relative_x, bins=NUM_BINS, range_min=0, range_max=SCREEN_WIDTH),
                    self.discretize(relative_y, bins=NUM_BINS, range_min=-SCREEN_HEIGHT, range_max=SCREEN_HEIGHT))
        else:
            return (NUM_BINS, NUM_BINS)

    def get_relative_enemy_bullet_position(self):
        threatening_bullets = [bullet for bullet in self.enemy_bullet_list if bullet.center_y < self.player.center_y + 200]
        if threatening_bullets:
            nearest_bullet = min(threatening_bullets, key=lambda b: abs(b.center_x - self.player.center_x))
            relative_x = nearest_bullet.center_x - self.player.center_x + SCREEN_WIDTH / 2
            relative_y = nearest_bullet.center_y - self.player.center_y
            return (self.discretize(relative_x, bins=NUM_BINS, range_min=0, range_max=SCREEN_WIDTH),
                    self.discretize(relative_y, bins=NUM_BINS, range_min=-SCREEN_HEIGHT, range_max=SCREEN_HEIGHT))
        else:
            return (NUM_BINS, NUM_BINS)

    def get_state(self):
        player_bin = self.discretize(self.player.center_x)
        asteroid_detected = self.detect_asteroids()
        enemy_detected = self.detect_enemies()
        bullet_detected = self.detect_enemy_bullets()
        enemy_rel_pos = self.get_relative_enemy_position()
        bullet_rel_pos = self.get_relative_enemy_bullet_position()
        state = (player_bin, asteroid_detected, enemy_detected, bullet_detected, enemy_rel_pos, bullet_rel_pos)
        return state

    def detect_asteroids(self):
        for asteroid in self.asteroid_list:
            if abs(asteroid.center_x - self.player.center_x) <= ASTEROID_DETECTION_RANGE:
                return 1
        return 0

    def detect_enemies(self):
        for enemy in self.enemy_list:
            if abs(enemy.center_x - self.player.center_x) <= ENEMY_DETECTION_RANGE:
                return 1
        return 0

    def detect_enemy_bullets(self):
        for bullet in self.enemy_bullet_list:
            if abs(bullet.center_x - self.player.center_x) <= ENEMY_BULLET_DETECTION_RANGE and bullet.center_y < self.player.center_y + 150:
                return 1
        return 0

    def choose_action(self):
        if random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        else:
            q_values = [self.q_table.get((self.state, a), 0) for a in range(NUM_ACTIONS)]
            return int(np.argmax(q_values))

    def perform_action(self, action):
        if action == 0:  # Se déplacer à gauche
            self.player.change_x = -PLAYER_SPEED
        elif action == 1:  # Se déplacer à droite
            self.player.change_x = PLAYER_SPEED
        elif action == 2:  # Tirer
            self.player.shoot(self.bullet_list)
        elif action == 3:  # Ne Rien Faire
            pass  # Aucune action n'est effectuée

    def update_q_table(self, next_state):
        q_current = self.q_table.get((self.state, self.last_action), 0)
        q_next = max([self.q_table.get((next_state, a), 0) for a in range(NUM_ACTIONS)])
        td_target = self.reward + DISCOUNT_FACTOR * q_next
        td_error = td_target - q_current
        self.q_table[(self.state, self.last_action)] = q_current + LEARNING_RATE * td_error

    def on_update(self, delta_time):
        if self.reset_required:
            self.setup()
            self.reset_required = False
            return

        self.state = self.get_state()
        self.last_action = self.choose_action()
        self.perform_action(self.last_action)

        self.player.update()
        self.player.update_cooldown()
        self.bullet_list.update()
        self.enemy_list.update()
        self.enemy_bullet_list.update()
        self.asteroid_list.update()

        self.enemy_shoot()


        # Pénalité pour tirer
        if self.last_action == 2:
            self.reward += ACTION_REWARD  # Pénalité plus élevée pour tirer
            if self.player.ammo <= 0 or not self.detect_enemies():
                self.reward += HIT_NOTHING_REWARD # Pénalité supplémentaire pour tir inutile

        # Collision des missiles du joueur avec les ennemis
        hit_enemy = False
        for bullet in self.bullet_list:
            hit_list = arcade.check_for_collision_with_list(bullet, self.enemy_list)
            if hit_list:
                hit_enemy = True
                bullet.remove_from_sprite_lists()
                for enemy in hit_list:
                    enemy.remove_from_sprite_lists()
                    self.score += 1
                    self.reward += HIT_ENEMIES_REWARD  # Récompense pour toucher un ennemi

        # Pénalité supplémentaire si le tir n'a touché aucun ennemi
        if self.last_action == 2 and not hit_enemy:
            self.reward += ACTION_REWARD  # Pénalité pour tir inefficace

        # Collision des missiles du joueur avec les astéroïdes
        for bullet in self.bullet_list:
            asteroid_hit_list = arcade.check_for_collision_with_list(bullet, self.asteroid_list)
            if asteroid_hit_list:
                bullet.remove_from_sprite_lists()
                for asteroid in asteroid_hit_list:
                    asteroid.take_damage()
                self.reward += HIT_ASTEROID_REWARD  # Pénalité plus élevée pour toucher un astéroïde

        # Collision des missiles ennemis avec les astéroïdes
        for bullet in self.enemy_bullet_list:
            asteroid_hit_list = arcade.check_for_collision_with_list(bullet, self.asteroid_list)
            if asteroid_hit_list:
                bullet.remove_from_sprite_lists()
                for asteroid in asteroid_hit_list:
                    asteroid.take_damage()

        # Collision des missiles ennemis avec le joueur
        for bullet in self.enemy_bullet_list:
            if arcade.check_for_collision(bullet, self.player):
                self.reward += LOOSE_REWARD
                self.game_over("Player hit by enemy bullet")
                return  # Terminer la mise à jour pour éviter des erreurs

        # Récompense pour éviter un astéroïde
        if self.detect_asteroids() and self.last_action in [0, 1]:
            self.reward += DODGE_ASTEROID_REWARD

        # Récompense pour se diriger vers l'ennemi le plus proche
        enemy_rel_x, _ = self.get_relative_enemy_position()
        if enemy_rel_x != NUM_BINS:
            if (enemy_rel_x < NUM_BINS // 2 and self.last_action == 0) or (enemy_rel_x > NUM_BINS // 2 and self.last_action == 1):
                self.reward += POSITION_REWARD

        # Récompense pour éviter un missile ennemi dangereux
        if self.detect_enemy_bullets() and self.last_action in [0, 1]:
            self.reward += DODGE_REWARD

        # Vérifier la fin de l'épisode si tous les ennemis sont vaincus
        if len(self.enemy_list) == 0:
            self.reward += WIN_REWARD  # Grande récompense pour gagner le jeu
            self.game_over("All enemies defeated")
            return  # Terminer la mise à jour

        # Augmenter la pénalité pour être à court de munitions
        if self.player.ammo <= 0 and len(self.enemy_list) > 0:
            self.reward += LOOSE_REWARD  # Pénalité plus sévère
            self.game_over("Out of ammunition")
            return

        next_state = self.get_state()
        self.update_q_table(next_state)
        self.state = next_state

    def on_draw(self):
        arcade.start_render()
        self.player.draw()
        self.bullet_list.draw()
        self.enemy_list.draw()
        self.enemy_bullet_list.draw()
        self.asteroid_list.draw()
        arcade.draw_text(f"Score: {self.score}", 10, 20, arcade.color.WHITE, 14)
        arcade.draw_text(f"Ammo: {self.player.ammo}", 10, 50, arcade.color.WHITE, 14)
        arcade.draw_text(f"Episode: {self.episode}", 10, 80, arcade.color.WHITE, 14)

    def enemy_shoot(self):
        for enemy in self.enemy_list:
            if random.random() < ENEMY_SHOOT_PROBABILITY:
                bullet = Bullet("images/enemy_bullet.png", 1)
                bullet.center_x = enemy.center_x
                bullet.top = enemy.bottom
                bullet.change_y = -BULLET_SPEED
                self.enemy_bullet_list.append(bullet)

    def save_q_table(self):
        with open("q_table.pkl", "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self):
        if os.path.exists("q_table.pkl"):
            with open("q_table.pkl", "rb") as f:
                self.q_table = pickle.load(f)
        else:
            self.q_table = {}


if __name__ == "__main__":
    game = SpaceInvadersGame()
    game.setup()
    arcade.run()
