import random
import torch
import torch.nn as nn
import torch.optim as optim
import arcade

from Core.Common import draw_grid
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
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Space Invaders - DQN")
        self.player = None
        self.bullet_list = None
        self.enemy_list = None
        self.enemy_bullet_list = None
        self.asteroid_list = None
        self.epsilon = None
        self.episode = None
        self.score = None
        self.reset_required = None
        self.total_reward = None
        self.last_action = None
        self.state = None
        self.memory = None
        self.history = None
        self.input_dim = None
        self.output_dim = None
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.batch_size = None
        self.gamma = None
        self.reward = None
        arcade.set_background_color(arcade.color.BLACK)
        self.set_update_rate(FRAME_RATE)
        self.initialize_game_state()
        self.initialize_dqn()
    def initialize_game_state(self):
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
        self.last_action = None
        self.state = None
        self.memory = []
        self.history = []

    def initialize_dqn(self):
        self.input_dim = 4  # Adjust as per state representation
        self.output_dim = len(Action)
        self.policy_net = DQN(self.input_dim, self.output_dim)
        self.target_net = DQN(self.input_dim, self.output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.batch_size = 32
        self.gamma = DISCOUNT_FACTOR

    def setup(self):
        self.initialize_game_state()
        self.initialize_player()
        self.initialize_enemies()

    def initialize_player(self):
        self.player = Player("images/player.png", 0.5)
        self.player.center_x = ((NUM_BINS // 2) - 1) * BIN_SIZE + BIN_SIZE / 2
        self.player.center_y = BIN_SIZE / 2
        self.player.ammo = NUM_ENEMY_ROWS * NUM_ENEMY_COLS + AMMO_EXTRA

    def initialize_enemies(self):
        for row in range(NUM_ENEMY_ROWS):
            for col in range(NUM_ENEMY_COLS):
                enemy = Enemy("images/enemy.png", 0.5)
                enemy.center_x = BIN_SIZE / 2 + col * BIN_SIZE
                enemy.center_y = SCREEN_HEIGHT - BIN_SIZE / 2 - (row * BIN_SIZE)
                enemy.change_x = ENEMY_SPEED
                self.enemy_list.append(enemy)

    def get_state(self):
        _, enemy_rel_pos = self.get_relative_position(self.enemy_list, ENEMY_DETECTION_RANGE)
        _, bullet_rel_pos = self.get_relative_position(self.enemy_bullet_list, ENEMY_BULLET_DETECTION_RANGE)
        return torch.tensor([*enemy_rel_pos, *bullet_rel_pos], dtype=torch.float32)

    def get_relative_position(self, obj_list, detection_range):
        filtered_objects = [obj for obj in obj_list if abs(obj.center_y - self.player.center_y) < detection_range]
        if filtered_objects:
            nearest_obj = min(filtered_objects, key=lambda obj: ((obj.center_x - self.player.center_x) ** 2 + (obj.center_y - self.player.center_y) ** 2) ** 0.5)
            relative_x = nearest_obj.center_x - self.player.center_x
            relative_y = nearest_obj.center_y - self.player.center_y
            return nearest_obj, (discretize(relative_x), discretize(relative_y))
        return None, (99, 99)

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

    def game_over(self, reason):
        print(f"Game Over: {reason}")
        print(f"Total Reward for Episode {self.episode}: {self.total_reward}")
        self.reset_required = True
        self.episode += 1
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self.history.append(self.total_reward)

    def on_update(self, delta_time):
        if self.reset_required:
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
        self.check_bullet_collisions()
        self.check_player_collisions()
        self.check_game_over_conditions()

    def check_bullet_collisions(self):
        for bullet in self.bullet_list:
            hit_list = arcade.check_for_collision_with_list(bullet, self.enemy_list)
            if hit_list:
                bullet.remove_from_sprite_lists()
                for enemy in hit_list:
                    enemy.remove_from_sprite_lists()
                    self.score += 1
                    self.reward += HIT_ENEMIES_REWARD

    def check_player_collisions(self):
        for bullet in self.enemy_bullet_list:
            if arcade.check_for_collision(bullet, self.player):
                self.reward += LOOSE_REWARD
                self.total_reward += self.reward
                self.game_over("Player hit by enemy bullet")
                return

    def check_game_over_conditions(self):
        if not self.enemy_list:
            self.reward += WIN_REWARD
            self.total_reward += self.reward
            self.game_over("All enemies defeated")

    def on_draw(self):
        arcade.start_render()
        draw_grid()
        self.highlight_targets()
        self.draw_game_objects()

    def highlight_targets(self):
        nearest_enemy, _ = self.get_relative_position(self.enemy_list, ENEMY_DETECTION_RANGE)
        nearest_bullet, _ = self.get_relative_position(self.enemy_bullet_list, ENEMY_BULLET_DETECTION_RANGE)
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