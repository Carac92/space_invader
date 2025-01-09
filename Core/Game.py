import os
import pickle
import random

import arcade
from matplotlib import pyplot as plt

from Entity.Bullet import Bullet
from Entity.Ennemy import Enemy
from Entity.Player import Player
from Setting import *


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
        self.total_reward = 0  # Variable pour accumuler les récompenses
        self.last_action = Action.DO_NOTHING
        self.state = None

        # Utiliser un dictionnaire pour la table Q
        self.q_table = {}
        self.history = []
        self.load_q_table()

        self.l, = plt.plot(self.history)

    def setup(self):
        self.player = Player("images/player.png", 0.5)
        self.player.center_x = (NUM_BINS//2) * BIN_SIZE + BIN_SIZE / 2
        self.player.center_y = BIN_SIZE / 2

        self.bullet_list = arcade.SpriteList()
        self.enemy_list = arcade.SpriteList()
        self.enemy_bullet_list = arcade.SpriteList()
        self.asteroid_list = arcade.SpriteList()

        for row in range(NUM_ENEMY_ROWS):
            for col in range(NUM_ENEMY_COLS):
                enemy = Enemy("images/enemy.png", 0.5)
                enemy.center_x = BIN_SIZE/2 + col * BIN_SIZE
                enemy.center_y = SCREEN_HEIGHT - BIN_SIZE/2 - (row * BIN_SIZE)
                enemy.change_x = ENEMY_SPEED
                self.enemy_list.append(enemy)

        """for col in range(8):
            asteroid = Asteroid("images/asteroid.png", 1, ASTEROID_LIFE)
            asteroid.center_x = 100 + col * 90
            asteroid.center_y = SCREEN_HEIGHT // 2
            self.asteroid_list.append(asteroid)"""

        total_enemies = NUM_ENEMY_ROWS * NUM_ENEMY_COLS
        self.player.ammo = total_enemies + AMMO_EXTRA

        self.score = 0
        self.reset_required = False
        self.total_reward = 0  # Réinitialisation du total des récompenses

    def game_over(self, reason):
        print(f"Game Over: {reason}")
        print(f"Total Reward for Episode {self.episode}: {self.total_reward}")  # Affichage du total des récompenses
        self.reset_required = True
        self.episode += 1
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self.history.append(self.total_reward)
        self.save_q_table()

    def discretize(self, value):
        bin_size = SCREEN_WIDTH // NUM_BINS
        index = int(value // bin_size)
        # prends en compte si le x est plutot vers la droite, ou plutot vers la gauche
        if value%bin_size > 0.5 * bin_size:
            index += 1
        return index

    """def get_relative_enemy_position(self):
        if self.enemy_list:
            nearest_enemy = min(self.enemy_list, key=lambda e: e.center_x - self.player.center_x)
            relative_x = nearest_enemy.center_x - self.player.center_x
            relative_y = nearest_enemy.center_y - self.player.center_y
            return (self.discretize(relative_x),
                    self.discretize(relative_y))
        else:
            return 99, 99"""

    def get_relative_enemy_position(self):
        if self.enemy_list:
            nearest_enemy = min(self.enemy_list, key=lambda e: ((e.center_x - self.player.center_x) ** 2 + (
                        e.center_y - self.player.center_y) ** 2) ** 0.5)
            relative_x = nearest_enemy.center_x - self.player.center_x
            relative_y = nearest_enemy.center_y - self.player.center_y
            return nearest_enemy, (self.discretize(relative_x), self.discretize(relative_y))
        else:
            return None, (99, 99)

    def get_relative_enemy_bullet_position(self):
        #pour ne prendre en compte que les bullets qui sont a moins de 200 pxl de hauteur du joueur
        threatening_bullets = [bullet for bullet in self.enemy_bullet_list if
                               abs(bullet.center_y - self.player.center_y) < 200]
        if threatening_bullets:
            #division euclidienne pour prendre en compte x et y
            nearest_bullet = min(threatening_bullets, key=lambda b: ((b.center_x - self.player.center_x) ** 2 + (
                        b.center_y - self.player.center_y) ** 2) ** 0.5)
            relative_x = nearest_bullet.center_x - self.player.center_x
            relative_y = nearest_bullet.center_y - self.player.center_y
            # Discrétiser les positions relatives
            relative_x_bin = self.discretize(relative_x)
            relative_y_bin = self.discretize(relative_y)
            return nearest_bullet,(relative_x_bin, relative_y_bin)
        else:
            return None,(99, 99)

    def get_state(self):
        player_bin = self.discretize(self.player.center_x)
        asteroid_detected = self.detect_asteroids()
        enemy_detected = self.detect_enemies()
        bullet_distance_bin = self.detect_enemy_bullets()
        _,enemy_rel_pos = self.get_relative_enemy_position()
        _,bullet_rel_pos = self.get_relative_enemy_bullet_position()
        #state = (player_bin, asteroid_detected, enemy_detected, bullet_distance_bin, enemy_rel_pos, bullet_rel_pos)
        state = (enemy_rel_pos,bullet_rel_pos)
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
        min_distance = ENEMY_BULLET_DETECTION_RANGE + 1
        for bullet in self.enemy_bullet_list:
            if bullet.center_y < self.player.center_y + 150:
                distance = bullet.center_x - self.player.center_x
                if abs(distance) <= ENEMY_BULLET_DETECTION_RANGE and abs(distance) < abs(min_distance):
                    min_distance = distance
        if abs(min_distance) <= ENEMY_BULLET_DETECTION_RANGE:
            # Discrétiser la distance en bins
            return self.discretize(min_distance)
        else:
            return 99  # Aucune menace immédiate

    """def choose_action(self):
        if random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        else:
            q_values = [self.q_table.get((self.state, a), 0) for a in range(NUM_ACTIONS)]
            return int(np.argmax(q_values))"""

    def choose_action(self):
        #enleve la possibilité de tirer si le cooldown n'est pas à 0
        if self.player.cooldown > 0:
            valid_actions = [action for action in Action if action != Action.SHOOT]
        else:
            valid_actions = list(Action)

        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = [self.q_table.get((self.state, action), 0) for action in valid_actions]
            max_value = max(q_values)
            #pour randomiser au cas où il y ai plusieurs actions avec un valeur similaire
            #par ex, les 4 à 0 car pas tester. Ca evite d'avoir toujours les memes actions
            max_actions = [action for action, q in zip(valid_actions, q_values) if q == max_value]
            return random.choice(max_actions)

    def perform_action(self, action):
        if action == Action.MOVE_LEFT:  # Utiliser Action.MOVE_LEFT au lieu de 0
            self.player.change_x = -PLAYER_SPEED
        elif action == Action.MOVE_RIGHT:  # Utiliser Action.MOVE_RIGHT au lieu de 1
            self.player.change_x = PLAYER_SPEED
        elif action == Action.SHOOT:  # Utiliser Action.SHOOT au lieu de 2
            self.player.shoot(self.bullet_list)
        elif action == Action.DO_NOTHING:  # Utiliser Action.DO_NOTHING au lieu de 3
            pass

    """def update_q_table(self, next_state):
        q_current = self.q_table.get((self.state, self.last_action), 0)
        q_next = max([self.q_table.get((next_state, a), 0) for a in range(NUM_ACTIONS)])
        td_target = self.reward + DISCOUNT_FACTOR * q_next
        td_error = td_target - q_current
        self.q_table[(self.state, self.last_action)] = q_current + LEARNING_RATE * td_error"""

    def update_q_table(self, next_state):
        if self.last_action == Action.SHOOT and self.player.cooldown > 0:
            return
        #recupere la valeur actuelle du state, 0 si elle n'existe pas
        q_current = self.q_table.get((self.state, self.last_action), 0)
        q_next = max([self.q_table.get((next_state, action), 0) for action in Action])
        td_target = self.reward + DISCOUNT_FACTOR * q_next
        td_error = td_target - q_current
        self.q_table[(self.state, self.last_action)] = q_current + LEARNING_RATE * td_error

    #A voir pour utiliser delta_time et limiter le nombre d update de la Q_table
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
        game_over_by_enemy_out_of_field = self.enemy_list.update()
        if game_over_by_enemy_out_of_field:
            self.game_over("Enemies reached the base")

        self.enemy_bullet_list.update()
        self.asteroid_list.update()

        self.enemy_shoot()

        self.reward = 0  # Réinitialisation de la récompense

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

        if self.last_action == Action.MOVE_LEFT or self.last_action == Action.MOVE_RIGHT:
            self.reward += ACTION_REWARD

        # Collision des missiles du joueur avec les astéroïdes
        for bullet in self.bullet_list:
            asteroid_hit_list = arcade.check_for_collision_with_list(bullet, self.asteroid_list)
            if asteroid_hit_list:
                bullet.remove_from_sprite_lists()
                for asteroid in asteroid_hit_list:
                    asteroid.take_damage()
                self.reward += HIT_ASTEROID_REWARD  # Pénalité pour toucher un astéroïde

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
                self.total_reward += self.reward  # Mise à jour du total des récompenses avant la fin
                self.game_over("Player hit by enemy bullet")
                return  # Terminer la mise à jour pour éviter des erreurs


        # Vérifier la fin de l'épisode si tous les ennemis sont vaincus
        if len(self.enemy_list) == 0:
            self.reward += WIN_REWARD  # Grande récompense pour gagner le jeu
            self.total_reward += self.reward  # Mise à jour du total des récompenses avant la fin
            self.game_over("All enemies defeated")
            return  # Terminer la mise à jour


        next_state = self.get_state()
        self.update_q_table(next_state)
        self.state = next_state

        self.total_reward += self.reward  # Mise à jour du total des récompenses

    def on_draw(self):
        arcade.start_render()

        # Dessiner la grille
        cell_size = SCREEN_WIDTH // NUM_BINS
        for x in range(0, SCREEN_WIDTH + 1, cell_size):
            arcade.draw_line(x, 0, x, SCREEN_HEIGHT, arcade.color.WHITE, 1)
        for y in range(0, SCREEN_HEIGHT + 1, cell_size):
            arcade.draw_line(0, y, SCREEN_WIDTH, y, arcade.color.WHITE, 1)

        #get nearest enemy
        nearest_enemy, _ = self.get_relative_enemy_position()
        #reset color of all enemies
        for enemy in self.enemy_list:
            enemy.color = arcade.color.WHITE
        #color nearest enemy in red
        if nearest_enemy:
            nearest_enemy.color = arcade.color.RED

        nearest_bullet, _ = self.get_relative_enemy_bullet_position()
        for bullet in self.enemy_bullet_list:
            bullet.color = arcade.color.WHITE
        if nearest_bullet:
            nearest_bullet.color = arcade.color.RED

        self.player.draw()
        self.bullet_list.draw()
        self.enemy_list.draw()
        self.enemy_bullet_list.draw()
        self.asteroid_list.draw()
        arcade.draw_text(f"Score: {self.score}", 10, 20, arcade.color.WHITE, 14)
        #arcade.draw_text(f"Ammo: {self.player.ammo}", 10, 50, arcade.color.WHITE, 14)
        arcade.draw_text(f"Episode: {self.episode}", 10, 80, arcade.color.WHITE, 14)
        arcade.draw_text(f"Total Reward: {self.total_reward}", 10, 110, arcade.color.WHITE, 14)
        arcade.draw_text(f"Epsilon(exploration rate): {self.epsilon}", 10, 50, arcade.color.WHITE, 14)


    def enemy_shoot(self):
        for enemy in self.enemy_list:
            if random.random() < ENEMY_SHOOT_PROBABILITY:
                bullet = Bullet("images/enemy_bullet.png", 1)
                bullet.center_x = enemy.center_x
                bullet.center_y = 549
                bullet.top = enemy.bottom
                bullet.change_y = -BULLET_SPEED
                self.enemy_bullet_list.append(bullet)

    def save_q_table(self):
        with open("q_table.pkl", "wb") as f:
            pickle.dump((self.q_table, self.history), f)

    def load_q_table(self):
        if os.path.exists("q_table.pkl"):
            with open("q_table.pkl", "rb") as f:
                self.q_table, self.history = pickle.load(f)
        else:
            self.q_table = {}
