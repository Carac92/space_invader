import random
import arcade
import numpy as np
from matplotlib import pyplot as plt

from Core.Q_Table_Utils import save_q_table, load_q_table
from Entity.Bullet import Bullet
from Entity.Ennemy import Enemy
from Entity.Player import Player
from Setting import *

class SpaceInvadersGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Space Invaders - IA avec Pénalités Ajustées")
        self.q_table = None
        self.history = None
        self.state = None
        self.last_action = None
        self.total_reward = None
        self.reward = None
        self.reset_required = None
        self.episode = None
        self.score = None
        self.epsilon = None
        self.asteroid_list = None
        self.enemy_bullet_list = None
        self.enemy_list = None
        self.bullet_list = None
        self.player = None
        arcade.set_background_color(arcade.color.BLACK)
        self.set_update_rate(FRAME_RATE)
        self.initialize_game_state()
        load_q_table()
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
        self.last_action = Action.DO_NOTHING
        self.state = None
        self.history = []
        self.q_table = {}

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

    def game_over(self, reason):
        print(f"Game Over: {reason}")
        print(f"Total Reward for Episode {self.episode}: {self.total_reward}")
        self.update_q_table(self.get_state())
        self.reset_required = True
        self.episode += 1
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self.history.append(self.total_reward)
        save_q_table(self.q_table, self.history)

    def reset(self):
        self.setup()

    def on_key_press(self, key, modifiers):
        if key == arcade.key.R:
            self.reset()
        elif key == arcade.key.Q:
            self.close_game()
        elif key == arcade.key.H:
            self.history = []

    def close_game(self):
        self.plot_rewards()
        arcade.close_window()

    def plot_rewards(self):
        window_size = 100
        smoothed_history = np.convolve(self.history, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smoothed_history)
        plt.xlabel("Épisode")
        plt.ylabel("Récompense moyenne")
        plt.title("Progression globale des Récompenses")
        plt.show()

    def get_relative_position(self, obj_list, detection_range):
        filtered_objects = [obj for obj in obj_list if abs(obj.center_y - self.player.center_y) < detection_range]
        if filtered_objects:
            nearest_obj = min(filtered_objects, key=lambda obj: ((obj.center_x - self.player.center_x) ** 2 + (obj.center_y - self.player.center_y) ** 2) ** 0.5)
            relative_x = nearest_obj.center_x - self.player.center_x
            relative_y = nearest_obj.center_y - self.player.center_y
            return nearest_obj, (discretize(relative_x), discretize(relative_y))
        return None, (99, 99)

    def get_state(self):
        _, enemy_rel_pos = self.get_relative_position(self.enemy_list, ENEMY_DETECTION_RANGE)
        _, bullet_rel_pos = self.get_relative_position(self.enemy_bullet_list, ENEMY_BULLET_DETECTION_RANGE)
        return enemy_rel_pos, bullet_rel_pos

    def choose_action(self):
        valid_actions = [action for action in Action if action != Action.SHOOT or self.player.cooldown == 0]
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        q_values = [self.q_table.get((self.state, action), 0) for action in valid_actions]
        max_value = max(q_values)
        max_actions = [action for action, q in zip(valid_actions, q_values) if q == max_value]
        return random.choice(max_actions)

    def perform_action(self, action):
        if action == Action.MOVE_LEFT:
            self.player.change_x = -PLAYER_SPEED
        elif action == Action.MOVE_RIGHT:
            self.player.change_x = PLAYER_SPEED
        elif action == Action.SHOOT:
            self.player.shoot(self.bullet_list)

    def update_q_table(self, next_state):
        if self.last_action == Action.SHOOT and self.player.cooldown > 0:
            return
        q_current = self.q_table.get((self.state, self.last_action), 0)
        q_next = max([self.q_table.get((next_state, action), 0) for action in Action])
        td_target = self.reward + DISCOUNT_FACTOR * q_next
        td_error = td_target - q_current
        self.q_table[(self.state, self.last_action)] = q_current + LEARNING_RATE * td_error

    def on_update(self, delta_time):
        if self.reset_required:
            self.reset()
            return

        self.state = self.get_state()
        self.last_action = self.choose_action()
        self.perform_action(self.last_action)
        self.update_game_objects()
        self.handle_collisions()
        next_state = self.get_state()
        self.update_q_table(next_state)
        self.state = next_state
        self.total_reward += self.reward

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


def draw_grid():
    cell_size = SCREEN_WIDTH // NUM_BINS
    for x in range(0, SCREEN_WIDTH + 1, cell_size):
        arcade.draw_line(x, 0, x, SCREEN_HEIGHT, arcade.color.WHITE, 1)
    for y in range(0, SCREEN_HEIGHT + 1, cell_size):
        arcade.draw_line(0, y, SCREEN_WIDTH, y, arcade.color.WHITE, 1)