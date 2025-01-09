# Dimensions de l'écran
from enum import Enum

class Action(Enum):
    DO_NOTHING = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    SHOOT = 3


SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
NUM_BINS = 10          # Nombre de bins pour la discrétisation de l'état
BIN_SIZE = SCREEN_WIDTH / NUM_BINS

# Joueur et mouvements
PLAYER_SPEED = 0
BULLET_SPEED = BIN_SIZE
BULLET_COOLDOWN = 10
NUM_ACTIONS = len(Action)

# Ennemis
ENEMY_SPEED = 0
ENEMY_DROP_SPEED = 0
ENEMY_SHOOT_PROBABILITY = 0.005
NUM_ENEMY_ROWS = 1
NUM_ENEMY_COLS = 5
ASTEROID_LIFE = 0

# Récompenses et pénalités
WIN_REWARD = 10000
LOOSE_REWARD = -10000

# Apprentissage par renforcement
LEARNING_RATE = 0.00025
DISCOUNT_FACTOR = 0.98
EPSILON = 1.0          # Taux d'exploration initial
EPSILON_MIN = 0.01     # Taux d'exploration minimum
EPSILON_DECAY = 0.99   # Facteur de décroissance d'EPSILON

# Détection
ASTEROID_DETECTION_RANGE = 100
ENEMY_DETECTION_RANGE = 200
ENEMY_BULLET_DETECTION_RANGE = 150

# Divers
AMMO_EXTRA = 10000000 # Détermine le nombre d'actions à partir de l'énumération Action


# Récompenses
HIT_ASTEROID_REWARD = 0
HIT_NOTHING_REWARD = 0
ACTION_REWARD = -1
DO_NOTHING_REWARD = 0
HIT_ENEMIES_REWARD = 10
DODGE_REWARD = 0
POSITION_REWARD = 0
DODGE_ASTEROID_REWARD = 0