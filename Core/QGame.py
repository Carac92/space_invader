
import os
import pickle
import random
from collections import deque

import numpy as np
import torch
from matplotlib import pyplot as plt

import arcade
from torch import optim, nn

from Entity.Bullet import Bullet
from Entity.Ennemy import Enemy
from Entity.Player import Player
from Setting import *

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.network(x)

class SpaceInvadersDQN(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Space Invaders - DQN")
        self.game_speed = FRAME_RATE
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
        self.batch_size = 64
        self.history = []
        # DQN specific
        self.input_size = 6  # State dimension (relative x,y for nearest enemy and bullet, player x, ammo)
        self.output_size = len(Action)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.input_size, self.output_size).to(self.device)
        self.target_net = DQN(self.input_size, self.output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = deque(maxlen=10000)

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
        next_state = self.get_state()
        self.update_q_table(next_state)
        print(
            f"player position: {self.discretize(self.player.center_x)}, {self.discretize(self.player.center_y)}")
        print(f"last action: {self.last_action}")
        print(f"Game Over: {reason}")
        print(
            f"Total Reward for Episode {self.episode}: {self.total_reward}")  # Affichage du total des récompenses
        self.reset_required = True
        self.episode += 1
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

    def on_key_press(self, key, modifiers):
        if key == arcade.key.R:
            self.reset()
        #Print q_table
        elif key == arcade.key.P:
            for _tuple in self.q_table.keys():
                print(str(_tuple), ":", self.q_table[_tuple])
        elif key == arcade.key.Q:
            self.close()
            self.plot_training_progress()
            exit(0)
        elif key == arcade.key.H:
            self.history = []
        elif key == arcade.key.UP:  # Accelerate game speed
            self.game_speed = max(1 / 4400, self.game_speed / 2)
            self.set_update_rate(self.game_speed)
        elif key == arcade.key.DOWN:  # Decelerate game speed
            self.game_speed = min(1 / 12, self.game_speed * 2)
            self.set_update_rate(self.game_speed)

    def plot_training_progress(self):
        """Affiche un graphique de progression des récompenses."""
        if len(self.history) < 10:
            print("Pas assez de données pour afficher un graphique.")
            return

        # Convertir l'historique en numpy array
        rewards = np.array(self.history)

        # Calculer la moyenne mobile (fenêtre de 100 épisodes ou moins si historique court)
        window_size = min(100, len(rewards))
        smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

        # Affichage du graphique
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(smoothed_rewards)), smoothed_rewards, label="Récompense Moyenne (lissée)", color="blue")
        plt.plot(range(len(rewards)), rewards, alpha=0.3, label="Récompenses Brutes", color="orange")
        plt.title("Progression de l'Apprentissage DQN")
        plt.xlabel("Épisodes")
        plt.ylabel("Récompense Totale")
        plt.legend()
        plt.grid()
        plt.show()

    def discretize(self, value):
        bin_size = SCREEN_WIDTH // NUM_BINS
        index = int(value // bin_size)
        # prends en compte si le x est plutot vers la droite, ou plutot vers la gauche
        if value % bin_size > 0.5 * bin_size:
            index += 1
        return index

    def get_relative_enemy_position(self):
        if self.enemy_list:
            nearest_enemy = min(self.enemy_list,
                                key=lambda e: ((e.center_x - self.player.center_x) ** 2 + (
                                        e.center_y - self.player.center_y) ** 2) ** 0.5)
            relative_x = nearest_enemy.center_x - self.player.center_x
            relative_y = nearest_enemy.center_y - self.player.center_y
            return nearest_enemy, (self.discretize(relative_x), self.discretize(relative_y))
        else:
            return None, (99, 99)

    def get_relative_enemy_bullet_position(self):
        # pour ne prendre en compte que les bullets qui sont a moins de ENEMY_BULLET_DETECTION_RANGE pxl de hauteur du joueur
        threatening_bullets = [bullet for bullet in self.enemy_bullet_list if
                               abs(bullet.center_y - self.player.center_y) < ENEMY_BULLET_DETECTION_RANGE]
        if threatening_bullets:
            # division euclidienne pour prendre en compte x et y
            nearest_bullet = min(threatening_bullets,
                                 key=lambda b: ((b.center_x - self.player.center_x) ** 2 + (
                                         b.center_y - self.player.center_y) ** 2) ** 0.5)
            relative_x = nearest_bullet.center_x - self.player.center_x
            relative_y = nearest_bullet.center_y - self.player.center_y
            # Discrétiser les positions relatives
            relative_x_bin = self.discretize(relative_x)
            relative_y_bin = self.discretize(relative_y)
            return nearest_bullet, (relative_x_bin, relative_y_bin)
        else:
            return None, (99, 99)

    def get_state(self):
        _, enemy_rel_pos = self.get_relative_enemy_position()
        _, bullet_rel_pos = self.get_relative_enemy_bullet_position()

        state = [
            enemy_rel_pos[0],  # Relative X position of nearest enemy
            enemy_rel_pos[1],  # Relative Y position of nearest enemy
            bullet_rel_pos[0],  # Relative X position of nearest bullet
            bullet_rel_pos[1],  # Relative Y position of nearest bullet
            self.discretize(self.player.center_x),  # Player X position
            self.player.ammo  # Remaining ammo
        ]

        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

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

    def choose_action(self):
         # Filtrer les actions valides
        if self.player.cooldown > 0:
            valid_actions = [action for action in Action if action != Action.SHOOT]
        else:
            valid_actions = list(Action)

        # Exploration : Choisir une action aléatoire
        if random.random() < self.epsilon:
            chosen_action = random.choice(valid_actions)
            return chosen_action

        # Exploitation : Utiliser le réseau de neurones pour choisir l'action optimale
        with torch.no_grad():
            state_tensor = self.get_state()
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
        # Filtrer les valeurs Q pour ne garder que celles des actions valides
        valid_q_values = [q_values[action.value] for action in valid_actions]
        # Trouver l'action avec la valeur Q maximale
        max_value = max(valid_q_values)
        tolerance = 1e-6
        max_actions = [action for action, q in zip(valid_actions, q_values) if abs(q - max_value) < tolerance]
        # Randomiser si plusieurs actions ont la même valeur maximale
        chosen_action = random.choice(max_actions)
        print(f"Exploitation: Action choisie via DQN -> {chosen_action}, Valeur Q -> {max_value}")
        return chosen_action

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
        if len(self.memory) < self.batch_size:
            return

            # Échantillonner un batch de transitions
        transitions = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch = zip(*transitions)

        state_batch = torch.cat(state_batch)
        action_batch = torch.tensor(action_batch, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float, device=self.device)
        next_state_batch = torch.cat(next_state_batch)

        # Q-values pour les actions prises
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Q-values cibles
        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (self.gamma * max_next_q_values)

        # Perte
        loss = nn.SmoothL1Loss()(q_values, target_q_values.unsqueeze(1))

        # Log de la perte
        print(f"Perte actuelle : {loss.item()}")

        # Optimisation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

    # A voir pour utiliser delta_time et limiter le nombre d update de la Q_table
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
        for enemy in self.enemy_list:
            if enemy.bottom + enemy.change_x < 0:
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

        # Chaque action fait perdre 1 point pour que l IA essaie de finir le plus vite possible
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
        if DISPLAY_MODE:
            arcade.start_render()

            # Dessiner la grille
            cell_size = SCREEN_WIDTH // NUM_BINS
            for x in range(0, SCREEN_WIDTH + 1, cell_size):
                arcade.draw_line(x, 0, x, SCREEN_HEIGHT, arcade.color.WHITE, 1)
            for y in range(0, SCREEN_HEIGHT + 1, cell_size):
                arcade.draw_line(0, y, SCREEN_WIDTH, y, arcade.color.WHITE, 1)

            # get nearest enemy
            nearest_enemy, _ = self.get_relative_enemy_position()
            # reset color of all enemies
            for enemy in self.enemy_list:
                enemy.color = arcade.color.WHITE
            # color nearest enemy in red
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
            # arcade.draw_text(f"Ammo: {self.player.ammo}", 10, 50, arcade.color.WHITE, 14)
            arcade.draw_text(f"Episode: {self.episode}", 10, 80, arcade.color.WHITE, 14)
            arcade.draw_text(f"Total Reward: {self.total_reward}", 10, 110, arcade.color.WHITE, 14)
            arcade.draw_text(f"Epsilon(exploration rate): {self.epsilon}", 10, 50, arcade.color.WHITE,
                             14)

    def enemy_shoot(self):
        for enemy in self.enemy_list:
            if random.random() < ENEMY_SHOOT_PROBABILITY:
                bullet = Bullet("images/enemy_bullet.png", 1)
                bullet.center_x = enemy.center_x
                bullet.center_y = enemy.center_y - BULLET_SPEED
                # bullet.top = enemy.bottom
                bullet.change_y = -BULLET_SPEED
                self.enemy_bullet_list.append(bullet)

    def save_q_table(self):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'history': self.history,
            'episode': self.episode,
        }, 'dqn_model.pth')

    def load_q_table(self):
        if os.path.exists('dqn_model.pth'):
            checkpoint = torch.load('dqn_model.pth')
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.history = checkpoint['history']
            self.episode = checkpoint['episode']
        else:
            print("No saved model found. Starting from scratch.")