import arcade

from Core.Game import SpaceInvadersGame
from Core.QGame import QSpaceInvadersGame

if __name__ == "__main__":
    game = SpaceInvadersGame()
    game.setup()
    arcade.run()

# if __name__ == "__main__":
#     game = QSpaceInvadersGame()
#     game.setup()
#     arcade.run()