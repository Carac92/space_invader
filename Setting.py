# Dimensions de l'écran
from enum import Enum

class Action(Enum):
    DO_NOTHING = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    SHOOT = 3

#Indique si le state prend en compte 1 missile ou 2 missiles
class StateNumberBullets(Enum):
    SINGLE = 1
    DOUBLE = 2

#True pour afficher les logs et l'interface graphique
DISPLAY_MODE = False

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
NUM_BINS = 10          # Nombre de bins pour la discrétisation de l'état
BIN_SIZE = SCREEN_WIDTH / NUM_BINS
FRAME_RATE = 1 / 1200

# Joueur et mouvements
PLAYER_SPEED = BIN_SIZE
BULLET_SPEED = BIN_SIZE
BULLET_COOLDOWN = 1
NUM_ACTIONS = len(Action)

# Ennemis
ENEMY_SPEED = BIN_SIZE
ENEMY_DROP_SPEED = BIN_SIZE
ENEMY_SHOOT_PROBABILITY = 0.025
NUM_ENEMY_ROWS = 2
NUM_ENEMY_COLS = 5
ASTEROID_LIFE = 0

# Apprentissage par renforcement
LEARNING_RATE = 0.95
DISCOUNT_FACTOR = 0.95
EPSILON = 0         # Taux d'exploration initial
EPSILON_MIN = 0  # Taux d'exploration minimum
EPSILON_DECAY = 0.99   # Facteur de décroissance d'EPSILON

# Détection
ASTEROID_DETECTION_RANGE = 100
ENEMY_DETECTION_RANGE = 200
ENEMY_BULLET_DETECTION_RANGE = 400

# Divers
AMMO_EXTRA = 10000000 # Détermine le nombre d'actions à partir de l'énumération Action


# Récompenses et pénalités
WIN_REWARD = 200
LOOSE_REWARD = -200

HIT_ASTEROID_REWARD = 0
HIT_NOTHING_REWARD = 0
ACTION_REWARD = -1
DO_NOTHING_REWARD = 0
HIT_ENEMIES_REWARD = 10
DODGE_REWARD = 0
POSITION_REWARD = 0
DODGE_ASTEROID_REWARD = 0