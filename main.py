import arcade
from Core.Game import SpaceInvadersGame
from Core.QGame import SpaceInvadersDQN


def main():
    print("Select the game mode:")
    print("1. Space Invaders - Qtable")
    print("2. Space Invaders with DQN")
    choice = input("Enter the number of your choice: ")

    if choice == "1":
        game = SpaceInvadersGame()
    elif choice == "2":
        game = SpaceInvadersDQN()
    else:
        print("Invalid choice. Exiting...")
        return

    game.setup()
    arcade.run()

if __name__ == "__main__":
    main()