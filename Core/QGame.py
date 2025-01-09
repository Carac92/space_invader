import os
import random
from collections import deque
import numpy as np
from matplotlib import pyplot as plt
import arcade
import torch
import torch.nn as nn
import torch.optim as optim

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

        # Game elements
        self.player = None
        self.bullet_list = None
        self.enemy_list = None
        self.enemy_bullet_list = None
        self.asteroid_list = None

        # DQN specific
        self.input_size = 6  # State dimension (relative x,y for nearest enemy and bullet, player x, ammo)
        self.output_size = len(Action)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.input_size, self.output_size).to(self.device)
        self.target_net = DQN(self.input_size, self.output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = deque(maxlen=10000)

        # Training parameters
        self.epsilon = EPSILON
        self.batch_size = 64
        self.gamma = DISCOUNT_FACTOR
        self.target_update = 10  # Update target network every N episodes

        # Game state
        self.episode = 0
        self.score = 0
        self.reset_required = False
        self.reward = 0
        self.total_reward = 0
        self.last_action = Action.DO_NOTHING
        self.state = None
        self.history = []

        self.load_model()

    def setup(self):
        self.player = Player("images/player.png", 0.5)
        self.player.center_x = SCREEN_WIDTH / 2
        self.player.center_y = BIN_SIZE / 2
        self.player.ammo = NUM_ENEMY_ROWS * NUM_ENEMY_COLS + AMMO_EXTRA

        self.bullet_list = arcade.SpriteList()
        self.enemy_list = arcade.SpriteList()
        self.enemy_bullet_list = arcade.SpriteList()

        for row in range(NUM_ENEMY_ROWS):
            for col in range(NUM_ENEMY_COLS):
                enemy = Enemy("images/enemy.png", 0.5)
                enemy.center_x = BIN_SIZE / 2 + col * BIN_SIZE
                enemy.center_y = SCREEN_HEIGHT - BIN_SIZE / 2 - row * BIN_SIZE
                enemy.change_x = ENEMY_SPEED
                self.enemy_list.append(enemy)

        self.score = 0
        self.total_reward = 0
        self.state = self.get_state_tensor()

    def get_state_tensor(self):
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

    def choose_action(self):
        if random.random() < self.epsilon:
            return random.choice(list(Action))

        with torch.no_grad():
            state = self.get_state_tensor()
            q_values = self.policy_net(state)
            return Action(q_values.argmax().item())

    def update_network(self):
        if len(self.memory) < self.batch_size:
            return

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

        # Optimisation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

    def on_update(self, delta_time):
        if self.reset_required:
            self.setup()
            self.reset_required = False
            return

        # Mise à jour des entités
        self.player.update()  # Assure-toi que cette ligne est bien présente
        self.player.update_cooldown()

        # Autres mises à jour
        self.bullet_list.update()
        self.enemy_list.update()
        self.enemy_bullet_list.update()

        # Gérer les actions IA
        current_state = self.get_state_tensor()
        action = self.choose_action()
        self.perform_action(action)

        # Gestion des collisions
        self.handle_collisions()

        # Mise à jour des transitions
        next_state = self.get_state_tensor()
        self.memory.append((current_state, action.value, self.reward, next_state))
        self.update_network()

        # Mise à jour de la récompense totale
        self.total_reward += self.reward

    def game_over(self, reason):
        print(f"Game Over: {reason}")
        print(f"Total Reward for Episode {self.episode}: {self.total_reward}")

        # Update target network periodically
        if self.episode % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.reset_required = True
        self.episode += 1
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self.history.append(self.total_reward)
        self.save_model()

    def handle_collisions(self):
        # Collisions entre les balles du joueur et les ennemis
        for bullet in self.bullet_list:
            hit_enemies = arcade.check_for_collision_with_list(bullet, self.enemy_list)
            if hit_enemies:
                bullet.remove_from_sprite_lists()
                for enemy in hit_enemies:
                    enemy.remove_from_sprite_lists()
                    self.reward += HIT_ENEMIES_REWARD
                    self.score += 1

        # Collision des balles ennemies avec le joueur
        for bullet in self.enemy_bullet_list:
            if arcade.check_for_collision(bullet, self.player):
                bullet.remove_from_sprite_lists()
                self.reward += LOOSE_REWARD
                self.game_over("Player hit by enemy bullet")
                return

        # Fin du jeu si tous les ennemis sont détruits
        if len(self.enemy_list) == 0:
            self.reward += WIN_REWARD
            self.game_over("All enemies defeated")

    def save_model(self):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'history': self.history,
            'episode': self.episode,
        }, 'dqn_model.pth')

    def load_model(self):
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

    def on_key_press(self, key, modifiers):
        if key == arcade.key.R:
            self.reset()
        elif key == arcade.key.Q:
            self.close()
            window_size = 100
            smoothed_history = np.convolve(self.history, np.ones(window_size) / window_size, mode='valid')
            plt.plot(smoothed_history)
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.title("DQN Training Progress")
            plt.show()
            exit(0)
        elif key == arcade.key.H:
            self.history = []
        elif key == arcade.key.UP:
            self.game_speed = max(1 / 4400, self.game_speed / 2)
            self.set_update_rate(self.game_speed)
        elif key == arcade.key.DOWN:
            self.game_speed = min(1 / 12, self.game_speed * 2)
            self.set_update_rate(self.game_speed)

    def discretize(self, value):
        """Convert continuous position to discrete bin index"""
        bin_size = SCREEN_WIDTH // NUM_BINS
        index = int(value // bin_size)
        if value % bin_size > 0.5 * bin_size:
            index += 1
        return index

    def get_relative_enemy_position(self):
        """Get the position of the nearest enemy relative to the player"""
        if self.enemy_list:
            nearest_enemy = min(self.enemy_list,
                                key=lambda e: ((e.center_x - self.player.center_x) ** 2 +
                                               (e.center_y - self.player.center_y) ** 2) ** 0.5)
            relative_x = nearest_enemy.center_x - self.player.center_x
            relative_y = nearest_enemy.center_y - self.player.center_y
            return nearest_enemy, (self.discretize(relative_x), self.discretize(relative_y))
        else:
            return None, (99, 99)

    def get_relative_enemy_bullet_position(self):
        """Get the position of the nearest enemy bullet relative to the player"""
        threatening_bullets = [bullet for bullet in self.enemy_bullet_list if
                               abs(bullet.center_y - self.player.center_y) < ENEMY_BULLET_DETECTION_RANGE]
        if threatening_bullets:
            nearest_bullet = min(threatening_bullets,
                                 key=lambda b: ((b.center_x - self.player.center_x) ** 2 +
                                                (b.center_y - self.player.center_y) ** 2) ** 0.5)
            relative_x = nearest_bullet.center_x - self.player.center_x
            relative_y = nearest_bullet.center_y - self.player.center_y
            return nearest_bullet, (self.discretize(relative_x), self.discretize(relative_y))
        else:
            return None, (99, 99)

    def perform_action(self, action):
        if action == Action.MOVE_LEFT:
            self.player.change_x = -PLAYER_SPEED
        elif action == Action.MOVE_RIGHT:
            self.player.change_x = PLAYER_SPEED
        elif action == Action.SHOOT:
            if self.player.cooldown == 0 and self.player.ammo > 0:
                self.player.shoot(self.bullet_list)
        elif action == Action.DO_NOTHING:
            self.player.change_x = 0

    def detect_asteroids(self):
        """Check if there are asteroids near the player"""
        for asteroid in self.asteroid_list:
            if abs(asteroid.center_x - self.player.center_x) <= ASTEROID_DETECTION_RANGE:
                return 1
        return 0

    def detect_enemies(self):
        """Check if there are enemies near the player"""
        for enemy in self.enemy_list:
            if abs(enemy.center_x - self.player.center_x) <= ENEMY_DETECTION_RANGE:
                return 1
        return 0

    def enemy_shoot(self):
        """Handle enemy shooting behavior"""
        for enemy in self.enemy_list:
            if random.random() < ENEMY_SHOOT_PROBABILITY:
                bullet = Bullet("images/enemy_bullet.png", 1)
                bullet.center_x = enemy.center_x
                bullet.center_y = enemy.center_y - BULLET_SPEED
                bullet.change_y = -BULLET_SPEED
                self.enemy_bullet_list.append(bullet)

    def reset(self):
        """Complete reset of the game and DQN parameters"""
        self.epsilon = EPSILON
        self.episode = 0
        self.score = 0
        self.reset_required = False
        self.reward = 0
        self.total_reward = 0
        self.last_action = Action.DO_NOTHING
        self.state = None
        self.history = []
        self.memory.clear()
        self.setup()

    def on_draw(self):
        """Render the game state"""
        if DISPLAY_MODE:
            arcade.start_render()

            # Draw grid
            cell_size = SCREEN_WIDTH // NUM_BINS
            for x in range(0, SCREEN_WIDTH + 1, cell_size):
                arcade.draw_line(x, 0, x, SCREEN_HEIGHT, arcade.color.WHITE, 1)
            for y in range(0, SCREEN_HEIGHT + 1, cell_size):
                arcade.draw_line(0, y, SCREEN_WIDTH, y, arcade.color.WHITE, 1)

            # Highlight nearest enemy and bullet
            nearest_enemy, _ = self.get_relative_enemy_position()
            for enemy in self.enemy_list:
                enemy.color = arcade.color.WHITE
            if nearest_enemy:
                nearest_enemy.color = arcade.color.RED

            nearest_bullet, _ = self.get_relative_enemy_bullet_position()
            for bullet in self.enemy_bullet_list:
                bullet.color = arcade.color.WHITE
            if nearest_bullet:
                nearest_bullet.color = arcade.color.RED

            # Draw game elements
            self.player.draw()
            self.bullet_list.draw()
            self.enemy_list.draw()
            self.enemy_bullet_list.draw()

            # Draw UI elements
            arcade.draw_text(f"Score: {self.score}", 10, 20, arcade.color.WHITE, 14)
            arcade.draw_text(f"Episode: {self.episode}", 10, 80, arcade.color.WHITE, 14)
            arcade.draw_text(f"Total Reward: {self.total_reward}", 10, 110, arcade.color.WHITE, 14)
            arcade.draw_text(f"Epsilon: {self.epsilon:.4f}", 10, 50, arcade.color.WHITE, 14)
