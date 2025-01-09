import random
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
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class QSpaceInvadersGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Space Invaders - IA avec Pénalités Ajustées")
        self.gamma = None
        self.batch_size = None
        self.memory = None
        self.optimizer = None
        self.target_net = None
        self.policy_net = None
        self.output_dim = None
        self.input_dim = None
        self.history = None
        self.state = None
        self.last_action = None
        self.total_reward = None
        self.reward = None
        self.reset_required = None
        self.score = None
        self.episode = None
        self.epsilon = None
        self.asteroid_list = None
        self.enemy_bullet_list = None
        self.enemy_list = None
        self.bullet_list = None
        self.player = None
        arcade.set_background_color(arcade.color.BLACK)
        self.set_update_rate(FRAME_RATE)
        self.init_game_state()
        self.init_dqn()
        self.display_mode = DISPLAY_MODE  # Nouveau attribut pour le mode d'affichage

    def init_game_state(self):
        self.player = None
        self.bullet_list = arcade.SpriteList()
        self.enemy_list = arcade.SpriteList()
        self.enemy_bullet_list = arcade.SpriteList()
        self.asteroid_list = arcade.SpriteList()
        self.epsilon = EPSILON
        self.episode = 0
        self.score = 0
        self.reset_required = False
        self.reward = 0
        self.total_reward = 0
        self.last_action = Action.DO_NOTHING
        self.state = None
        self.history = []

    def init_dqn(self):
        self.input_dim = 4  # Example state dimensions (adjust as needed)
        self.output_dim = len(Action)
        self.policy_net = DQN(self.input_dim, self.output_dim)
        self.target_net = DQN(self.input_dim, self.output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = []
        self.batch_size = 32
        self.gamma = DISCOUNT_FACTOR

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_net(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def setup(self):
        self.init_game_state()
        self.player = Player("images/player.png", 0.5)
        self.player.center_x = ((NUM_BINS // 2) - 1) * BIN_SIZE + BIN_SIZE / 2
        self.player.center_y = BIN_SIZE / 2

        self.init_enemies()
        self.player.ammo = len(self.enemy_list) + AMMO_EXTRA

    def init_enemies(self):
        for row in range(NUM_ENEMY_ROWS):
            for col in range(NUM_ENEMY_COLS):
                enemy = Enemy("images/enemy.png", 0.5)
                enemy.center_x = BIN_SIZE / 2 + col * BIN_SIZE
                enemy.center_y = SCREEN_HEIGHT - BIN_SIZE / 2 - (row * BIN_SIZE)
                enemy.change_x = ENEMY_SPEED
                self.enemy_list.append(enemy)

    def game_over(self, reason):
        print(f"Game Over: {reason}")
        print(f"Total Reward for Episode {self.episode}: {self.total_reward}")
        self.reset_required = True
        self.episode += 1
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self.history.append(self.total_reward)

    def reset(self):
        self.setup()

    def on_key_press(self, key, modifiers):
        if key == arcade.key.R:
            self.reset()
        elif key == arcade.key.Q:
            self.close_game()
        elif key == arcade.key.D:  # Nouveau bouton pour afficher/masquer l'affichage
            self.display_mode = not self.display_mode

    def close_game(self):
        self.plot_rewards()
        exit(0)

    def plot_rewards(self):
        window_size = 100
        smoothed_history = np.convolve(self.history, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smoothed_history)
        plt.xlabel("Épisode")
        plt.ylabel("Récompense moyenne")
        plt.title("Progression globale des Récompenses")
        plt.show()

    def get_state(self):
        _, enemy_rel_pos = self.get_relative_enemy_position()
        _, bullet_rel_pos = self.get_relative_enemy_bullet_position()
        return torch.tensor([*enemy_rel_pos, *bullet_rel_pos], dtype=torch.float32)

    def get_relative_enemy_position(self):
        if not self.enemy_list:
            return None, (99, 99)
        nearest_enemy = min(self.enemy_list, key=lambda e: self.euclidean_distance(e, self.player))
        relative_x = nearest_enemy.center_x - self.player.center_x
        relative_y = nearest_enemy.center_y - self.player.center_y
        return nearest_enemy, (discretize(relative_x), discretize(relative_y))

    def get_relative_enemy_bullet_position(self):
        threatening_bullets = [bullet for bullet in self.enemy_bullet_list
                               if abs(bullet.center_y - self.player.center_y) < 200]
        if not threatening_bullets:
            return None, (99, 99)
        nearest_bullet = min(threatening_bullets, key=lambda b: self.euclidean_distance(b, self.player))
        relative_x = nearest_bullet.center_x - self.player.center_x
        relative_y = nearest_bullet.center_y - self.player.center_y
        return nearest_bullet, (discretize(relative_x), discretize(relative_y))

    def euclidean_distance(self, obj1, obj2):
        return ((obj1.center_x - obj2.center_x) ** 2 + (obj1.center_y - obj2.center_y) ** 2) ** 0.5

    def choose_action(self):
        if random.random() < self.epsilon:
            return random.choice(range(self.output_dim))
        with torch.no_grad():
            state = self.get_state().unsqueeze(0)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def perform_action(self, action):
        if action == Action.MOVE_LEFT.value:
            self.player.change_x = -PLAYER_SPEED
        elif action == Action.MOVE_RIGHT.value:
            self.player.change_x = PLAYER_SPEED
        elif action == Action.SHOOT.value:
            self.player.shoot(self.bullet_list)

    def on_update(self, delta_time):
        if self.reset_required:
            self.reset_required = False
            self.setup()
            return

        state = self.get_state()
        action = self.choose_action()
        self.perform_action(action)

        self.update_game_objects()
        self.handle_collisions()

        next_state = self.get_state()
        reward = self.reward
        done = len(self.enemy_list) == 0 or self.reset_required

        self.remember(state.numpy(), action, reward, next_state.numpy(), done)
        self.replay()
        if self.episode % 10 == 0:
            self.update_target_network()

        self.total_reward += reward

    def update_game_objects(self):
        self.player.update()
        self.player.update_cooldown()
        self.bullet_list.update()
        self.enemy_list.update()
        self.enemy_bullet_list.update()
        self.asteroid_list.update()
        self.enemy_shoot()

    def handle_collisions(self):
        self.reward = ACTION_REWARD
        self.handle_bullet_collisions()
        self.handle_enemy_collisions()
        self.handle_asteroid_collisions()
        self.handle_player_collisions()

    def handle_bullet_collisions(self):
        for bullet in self.bullet_list:
            hit_list = arcade.check_for_collision_with_list(bullet, self.enemy_list)
            if hit_list:
                bullet.remove_from_sprite_lists()
                for enemy in hit_list:
                    enemy.remove_from_sprite_lists()
                    self.score += 1
                    self.reward += HIT_ENEMIES_REWARD

    def handle_enemy_collisions(self):
        for enemy in self.enemy_list:
            if enemy.bottom + enemy.change_x < 0:
                self.reward += LOOSE_REWARD
                self.total_reward += self.reward
                self.game_over("Enemies reached the base")
                return

    def handle_asteroid_collisions(self):
        for bullet in self.bullet_list:
            asteroid_hit_list = arcade.check_for_collision_with_list(bullet, self.asteroid_list)
            if asteroid_hit_list:
                bullet.remove_from_sprite_lists()
                for asteroid in asteroid_hit_list:
                    asteroid.take_damage()
                    self.reward += HIT_ASTEROID_REWARD

    def handle_player_collisions(self):
        for bullet in self.enemy_bullet_list:
            if arcade.check_for_collision(bullet, self.player):
                self.reward += LOOSE_REWARD
                self.total_reward += self.reward
                self.game_over("Player hit by enemy bullet")
                return

        if not self.enemy_list:
            self.reward += WIN_REWARD
            self.total_reward += self.reward
            self.game_over("All enemies defeated")

    def on_draw(self):
        if self.display_mode:
            arcade.start_render()
            self.draw_grid()
            self.highlight_targets()
            self.draw_game_objects()

    def draw_grid(self):
        cell_size = SCREEN_WIDTH // NUM_BINS
        for x in range(0, SCREEN_WIDTH + 1, cell_size):
            arcade.draw_line(x, 0, x, SCREEN_HEIGHT, arcade.color.WHITE, 1)
        for y in range(0, SCREEN_HEIGHT + 1, cell_size):
            arcade.draw_line(0, y, SCREEN_WIDTH, y, arcade.color.WHITE, 1)

    def highlight_targets(self):
        nearest_enemy, _ = self.get_relative_enemy_position()
        nearest_bullet, _ = self.get_relative_enemy_bullet_position()

        for enemy in self.enemy_list:
            enemy.color = arcade.color.WHITE
        if nearest_enemy:
            nearest_enemy.color = arcade.color.RED

        for bullet in self.enemy_bullet_list:
            bullet.color = arcade.color.WHITE
        if nearest_bullet:
            nearest_bullet.color = arcade.color.RED

    def draw_game_objects(self):
        self.player.draw()
        self.bullet_list.draw()
        self.enemy_list.draw()
        self.enemy_bullet_list.draw()
        self.asteroid_list.draw()
        arcade.draw_text(f"Score: {self.score}", 10, 20, arcade.color.WHITE, 14)
        arcade.draw_text(f"Episode: {self.episode}", 10, 80, arcade.color.WHITE, 14)
        arcade.draw_text(f"Total Reward: {self.total_reward}", 10, 110, arcade.color.WHITE, 14)
        arcade.draw_text(f"Epsilon (exploration rate): {self.epsilon}", 10, 50, arcade.color.WHITE, 14)

    def enemy_shoot(self):
        for enemy in self.enemy_list:
            if random.random() < ENEMY_SHOOT_PROBABILITY:
                bullet = Bullet("images/enemy_bullet.png", 1)
                bullet.center_x = enemy.center_x
                bullet.center_y = enemy.center_y - BULLET_SPEED
                bullet.change_y = -BULLET_SPEED
                self.enemy_bullet_list.append(bullet)


def discretize(value):
    bin_size = SCREEN_WIDTH // NUM_BINS
    index = int(value // bin_size)
    if value % bin_size > 0.5 * bin_size:
        index += 1
    return index
