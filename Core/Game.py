import os
import pickle
import random

import numpy as np
from matplotlib import pyplot as plt

import arcade

from Entity.Bullet import Bullet
from Entity.Ennemy import Enemy
from Entity.Player import Player
from Setting import *


class SpaceInvadersGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Space Invaders - IA avec Pénalités Ajustées")
        self.game_speed = FRAME_RATE
        self.display_mode = DISPLAY_MODE
        arcade.set_background_color(arcade.color.BLACK)
        self.set_update_rate(self.game_speed)
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
        self.game_mode = GAME_MODE

    def setup(self):
        self.player = Player("images/player.png", 0.5)
        self.player.center_x = ((NUM_BINS // 2) - 1) * BIN_SIZE + BIN_SIZE / 2
        self.player.center_y = BIN_SIZE / 2

        self.bullet_list = arcade.SpriteList()
        self.enemy_list = arcade.SpriteList()
        self.enemy_bullet_list = arcade.SpriteList()
        self.asteroid_list = arcade.SpriteList()

        for row in range(NUM_ENEMY_ROWS):
            for col in range(NUM_ENEMY_COLS):
                enemy = Enemy("images/enemy.png", 0.5)
                enemy.center_x = BIN_SIZE / 2 + col * BIN_SIZE
                enemy.center_y = SCREEN_HEIGHT - BIN_SIZE / 2 - (row * BIN_SIZE)
                enemy.change_x = ENEMY_SPEED
                self.enemy_list.append(enemy)

        total_enemies = NUM_ENEMY_ROWS * NUM_ENEMY_COLS
        self.player.ammo = total_enemies + AMMO_EXTRA

        self.score = 0
        self.reset_required = False
        self.total_reward = 0  # Réinitialisation du total des récompenses

    def game_over(self, reason):
        if self.display_mode:
            print(f"player position: {self.discretize(self.player.center_x)}, {self.discretize(self.player.center_y)}")
            print(f"last action: {self.last_action}")
            print(f"Game Over: {reason}")
            print(f"Total Reward for Episode {self.episode}: {self.total_reward}")  # Affichage du total des récompenses
        self.reset_required = True
        self.episode += 1
        if self.game_mode == GameMode.AI_MODE:
            next_state = self.get_state()
            self.update_q_table(next_state)
            self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
            self.history.append(self.total_reward)
            self.save_q_table()

    def reset(self):
        self.epsilon = EPSILON
        self.episode = 0
        self.score = 0
        self.reset_required = False
        self.reward = 0
        self.total_reward = 0  # Variable pour accumuler les récompenses
        self.last_action = Action.DO_NOTHING
        self.state = None
        self.history = []
        self.q_table = {}
        self.setup()

    """
    R: Reset history et q_table
    H: Reset history
    P: Print la q-table (1 fois)
    D: Active/Desactive les logs et l'affichage graphique
    UP: Accelere le game speed par 2
    DOWN: Divise le game speed par 2
    SPACE: Passe du mode Player au mode IA ou inversement
    LEFT: (Player mode) Deplace le vaisseau sur la gauche
    RIGHT: (Player mode) Deplace le vaisseau sur la droite
    Z: (Player mode) Tire un missile si le cooldown est à 0
    """

    def on_key_press(self, key, modifiers):
        #Reset history et q_table
        if key == arcade.key.R:
            self.reset()
        #Print q_table 1 fois
        elif key == arcade.key.P:
            for _tuple in self.q_table.keys():
                print(str(_tuple), ":", self.q_table[_tuple])
        #Quitte le jeu
        elif key == arcade.key.Q:
            self.close()
            window_size = 100

            # Moyenne mobile
            smoothed_history = np.convolve(self.history, np.ones(window_size) / window_size,
                                           mode='valid')

            # Affichage de la courbe lissée
            plt.plot(smoothed_history)
            plt.xlabel("Épisode")
            plt.ylabel("Récompense moyenne")
            plt.title("Progression globale des Récompenses")
            plt.show()
            exit(0)
        #Efface l'history
        elif key == arcade.key.H:
            self.history = []
        #Active ou desactive l'affichage
        elif key == arcade.key.D:
            self.display_mode = not self.display_mode
            arcade.start_render()
        # Accelere game speed
        elif key == arcade.key.UP:
            self.game_speed = max(1 / 4400, self.game_speed / 2)
            self.set_update_rate(self.game_speed)
        # Decelere game speed
        elif key == arcade.key.DOWN:
            self.game_speed = min(1 / 12, self.game_speed * 2)
            self.set_update_rate(self.game_speed)
        #Si en mode Player, débloque les touches de déplacements et de tir
        if self.game_mode == GameMode.HUMAN_MODE:
            if key == arcade.key.LEFT:
                self.perform_action(Action.MOVE_LEFT)
            if key == arcade.key.RIGHT:
                self.perform_action(Action.MOVE_RIGHT)
            if key == arcade.key.Z:
                self.perform_action(Action.SHOOT)
        #Passe du mode Player au mode IA et inversement
        if key == arcade.key.SPACE:
            self.game_mode = GameMode.HUMAN_MODE if self.game_mode == GameMode.AI_MODE else GameMode.AI_MODE

    def discretize(self, value):
        bin_size = SCREEN_WIDTH // NUM_BINS
        index = int(value // bin_size)
        # prends en compte si le x est plutot vers la droite, ou plutot vers la gauche
        if value%bin_size > 0.5 * bin_size:
            index += 1
        return index

    def get_relative_enemy_position(self):
        if self.enemy_list:
            #utilise la distance euclidienne pour définir l'ennemi le plus proche
            nearest_enemy = min(self.enemy_list, key=lambda e: ((e.center_x - self.player.center_x) ** 2 + (
                        e.center_y - self.player.center_y) ** 2) ** 0.5)
            relative_x = nearest_enemy.center_x - self.player.center_x
            relative_y = nearest_enemy.center_y - self.player.center_y
            return nearest_enemy, (self.discretize(relative_x), self.discretize(relative_y))
        else:
            return None, (99, 99)

    def get_relative_enemy_bullet_position(self):
        #pour ne prendre en compte que les bullets qui sont a moins de ENEMY_BULLET_DETECTION_RANGE pxl de hauteur du joueur
        threatening_bullets = [bullet for bullet in self.enemy_bullet_list if
                               abs(bullet.center_y - self.player.center_y) < ENEMY_BULLET_DETECTION_RANGE]
        if threatening_bullets:
            #division euclidienne pour prendre en compte x et y
            nearest_bullet = min(threatening_bullets, key=lambda b: ((b.center_x - self.player.center_x) ** 2 + (
                        b.center_y - self.player.center_y) ** 2) ** 0.5)
            relative_x = nearest_bullet.center_x - self.player.center_x
            relative_y = nearest_bullet.center_y - self.player.center_y
            return nearest_bullet,(self.discretize(relative_x), self.discretize(relative_y))
        else:
            return None,(99, 99)

    def get_relative_two_enemy_bullet_position(self):
        closest_bullets = []
        #pour ne prendre en compte que les bullets qui sont a moins de ENEMY_BULLET_DETECTION_RANGE pxl de hauteur du joueur
        threatening_bullets = [bullet for bullet in self.enemy_bullet_list if
                               abs(bullet.center_y - self.player.center_y) < ENEMY_BULLET_DETECTION_RANGE]
        if threatening_bullets:
            # Trier les balles par leur distance au joueur, en ordre croissant
            sorted_bullets = sorted(threatening_bullets, key=lambda b: ((b.center_x - self.player.center_x) ** 2 + (
                    b.center_y - self.player.center_y) ** 2) ** 0.5)

            # Vérifiez s'il y a au moins deux balles pour obtenir la deuxième la plus proche
            if len(sorted_bullets) > 1:
                nearest_bullet = sorted_bullets[0]  # La balle la plus proche
                second_nearest_bullet = sorted_bullets[1]  # La deuxième plus proche
            else:
                nearest_bullet = sorted_bullets[0] if sorted_bullets else None
                second_nearest_bullet = None

            relative_nearest_x = nearest_bullet.center_x - self.player.center_x
            relative_nearest_y = nearest_bullet.center_y - self.player.center_y
            relative_second_nearest_x = (second_nearest_bullet.center_x - self.player.center_x) if second_nearest_bullet else None
            relative_second_nearest_y = (second_nearest_bullet.center_y - self.player.center_y) if second_nearest_bullet else None
            discretize_relative_second_nearest_x = self.discretize(relative_second_nearest_x) if relative_second_nearest_x else 99
            discretize_relative_second_nearest_y = self.discretize(relative_second_nearest_y) if relative_second_nearest_y else 99
            return nearest_bullet, second_nearest_bullet,(self.discretize(relative_nearest_x), self.discretize(relative_nearest_y)),(discretize_relative_second_nearest_x, discretize_relative_second_nearest_y)

        else:
            return None,None,(99, 99),(99, 99)

    def get_state(self):
        _,enemy_rel_pos = self.get_relative_enemy_position()
        if LEARNING_MODE == StateNumberBullets.SINGLE:
            _,bullet_rel_pos = self.get_relative_enemy_bullet_position()
            return enemy_rel_pos,bullet_rel_pos
        elif LEARNING_MODE == StateNumberBullets.DOUBLE:
            _,_,closest_bullet_rel_pos,second_closest_bullet_rel_pos = self.get_relative_two_enemy_bullet_position()
            state = (enemy_rel_pos,closest_bullet_rel_pos,second_closest_bullet_rel_pos)
            return state

    def detect_asteroids(self):
        for asteroid in self.asteroid_list:
            if abs(asteroid.center_x - self.player.center_x) <= ASTEROID_DETECTION_RANGE:
                return 1
        return 0

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
            #print("tab: " + str(q_values))
            max_value = max(q_values)
            #print("max: " + str(max_value))
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

    def update_q_table(self, next_state):
        if self.game_mode == GameMode.HUMAN_MODE:
            return
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

        if self.game_mode == GameMode.AI_MODE:
            self.state = self.get_state()
            self.last_action = self.choose_action()
            self.perform_action(self.last_action)

        self.player.update()
        self.player.update_cooldown()
        self.bullet_list.update()
        for enemy in self.enemy_list:
            if enemy.center_x + enemy.change_x < 0 or enemy.center_x + enemy.change_x > NUM_BINS -1:
                if enemy.center_y - ENEMY_DROP_SPEED < 0:
                    self.reward += LOOSE_REWARD
                    self.total_reward += self.reward  # Mise à jour du total des récompenses avant la fin
                    self.game_over("Enemies reached the base")
        self.enemy_list.update()

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

        #Chaque action fait perdre 1 point pour que l IA essaie de finir le plus vite possible
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
        if self.display_mode:
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

            if LEARNING_MODE == StateNumberBullets.SINGLE:
                nearest_bullet, _ = self.get_relative_enemy_bullet_position()
                for bullet in self.enemy_bullet_list:
                    bullet.color = arcade.color.WHITE
                if nearest_bullet:
                    nearest_bullet.color = arcade.color.RED
            elif LEARNING_MODE == StateNumberBullets.DOUBLE:
                nearest_bullet, second_nearest_bullet, _, _ = self.get_relative_two_enemy_bullet_position()
                for bullet in self.enemy_bullet_list:
                    bullet.color = arcade.color.WHITE
                if nearest_bullet:
                    nearest_bullet.color = arcade.color.RED
                if second_nearest_bullet:
                    second_nearest_bullet.color = arcade.color.RED

            self.player.draw()
            self.bullet_list.draw()
            self.enemy_list.draw()
            self.enemy_bullet_list.draw()
            self.asteroid_list.draw()
            arcade.draw_text(f"FPS: {1/self.game_speed}", 10, 20, arcade.color.WHITE, 14)
            #arcade.draw_text(f"Ammo: {self.player.ammo}", 10, 50, arcade.color.WHITE, 14)
            arcade.draw_text(f"Episode: {self.episode}", 10, 80, arcade.color.WHITE, 14)
            arcade.draw_text(f"Total Reward: {self.total_reward}", 10, 110, arcade.color.WHITE, 14)
            arcade.draw_text(f"Epsilon(exploration rate): {self.epsilon}", 10, 50, arcade.color.WHITE, 14)


    def enemy_shoot(self):
        for enemy in self.enemy_list:
            if random.random() < ENEMY_SHOOT_PROBABILITY:
                bullet = Bullet("images/enemy_bullet.png", 1)
                bullet.center_x = enemy.center_x
                bullet.center_y = enemy.center_y - BULLET_SPEED
                #bullet.top = enemy.bottom
                bullet.change_y = -BULLET_SPEED
                self.enemy_bullet_list.append(bullet)

    def save_q_table(self):
        with open("q_table.pkl", "wb") as f:
            pickle.dump((self.q_table, self.history), f)

    def load_q_table(self):
        if os.path.exists("q_table.pkl"):
            with open("q_table.pkl", "rb") as f:
                self.q_table, self.history = pickle.load(f)
            """for tuple in self.q_table.keys():
                if tuple[0][0][1] == 1:
                    print(
                        "closest E: " + str(tuple[0][0]) + ", " +
                        "closest B: " + str(tuple[0][0]) + ", " +
                        "Action: " + str(tuple[1]) + ", " +
                        "Reward: " + str(self.q_table[tuple])
                    )"""
        else:
            self.q_table = {}
