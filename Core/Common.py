import arcade

from Setting import *

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