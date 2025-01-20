import arcade
from Core.Game import SpaceInvadersGame
from Core.QGame import SpaceInvadersDQN

menu_art = """

  _________                           .___                         .___            
 /   _____/__________    ____  ____   |   | _______  _______     __| _/___________ 
 \_____  \\____ \__  \ _/ ___\/ __ \  |   |/    \  \/ /\__  \   / __ |/ __ \_  __ \
 /        \  |_> > __ \\  \__\  ___/  |   |   |  \   /  / __ \_/ /_/ \  ___/|  | \/
/_______  /   __(____  /\___  >___  > |___|___|  /\_/  (____  /\____ |\___  >__|   
        \/|__|       \/     \/    \/           \/           \/      \/    \/       

"""

menu_options = [
    "1. Start Game with Q-learning",
    "2. Start Game with Deep-learning",
    "3. Quit"
]


def display_menu():
    print(menu_art)
    print("\nMain Menu:")
    for option in menu_options:
        print(option)


def main():
    global game
    while True:
        display_menu()
        choice = input("\nSelect an option (1-3): ").strip()

        if choice == "1":
            print("\nStarting new game with Q-learning...")
            game = SpaceInvadersGame()
            game.setup()
            arcade.run()
        elif choice == "2":
            print("\nStarting new game with Deep-learning...")
            game = SpaceInvadersDQN()
            game.setup()
            arcade.run()
        elif choice == "3":
            print("\nExiting. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please select a valid option.")


if __name__ == "__main__":
    main()
