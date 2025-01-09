# Constantes
from Actions import Action

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
NUM_BINS = 10  # Réduction du nombre de bins pour gérer la taille de l'espace d'état
BIN_SIZE = SCREEN_WIDTH / NUM_BINS
PLAYER_SPEED = BIN_SIZE
BULLET_SPEED = BIN_SIZE
ENEMY_SPEED = 0
ENEMY_DROP_SPEED = BIN_SIZE
NUM_ENEMY_ROWS = 2
NUM_ENEMY_COLS = 5
BULLET_COOLDOWN = 10
ENEMY_SHOOT_PROBABILITY = 0.005
ASTEROID_LIFE = 0
AMMO_EXTRA = 10000000
NUM_ACTIONS = len(Action)  # Détermine le nombre d'actions à partir de l'énumération Action
  # Ajout de l'action "Ne Rien Faire"
LEARNING_RATE = 0.95  # Taux d'apprentissage ajusté
DISCOUNT_FACTOR = 0.98  # Facteur de décote ajusté
EPSILON = 0.00  # Commence avec une exploration maximale
EPSILON_MIN = 0.00
EPSILON_DECAY = 0.99  # Décroissance plus lente de l'exploration

# Portées des zones de détection
ASTEROID_DETECTION_RANGE = 100
ENEMY_DETECTION_RANGE = 200
ENEMY_BULLET_DETECTION_RANGE = 150

# Récompenses
HIT_ASTEROID_REWARD = 0
HIT_NOTHING_REWARD = 0
ACTION_REWARD = -1
DO_NOTHING_REWARD = 0
HIT_ENEMIES_REWARD = 10
LOOSE_REWARD = -100
WIN_REWARD = 100
DODGE_REWARD = 0
POSITION_REWARD = 0
DODGE_ASTEROID_REWARD = 0