"""
Params for Genetic Algorithm
"""

# Define which data set you are using
#DATA_SET = 'Yang_sim'
#DATA_SET = 'Chen'
DATA_SET = 'Peurifoy'
TEST_RATIO = 0.2

# GA SPECIFIC PARAMETERS
# Essential Params
POP_SIZE = 40
ELITISM = 1
MUTATION = 0.05
CROSSOVER = 0.8
K = 5

# Categorical Params
SELECT_OPS = 'roulette' # 'decimation' 'tournament'
CROSS_OPS = 'single-point' # 'uniform'
GA_EVAL = True # Geometry -> Spectra calculation done by simulator function rather than a neural network

# Optimization Params
EVAL_STEP = 20
GENERATIONS = 300
STOP_THRESHOLD = 1e-9

# Data specific Params
X_RANGE = range(2,16)#[i for i in range(2, 16 )]
#Y_RANGE = [i for i in range(10 , 2011 )]                       # Real Meta-material dataset range
Y_RANGE = range(16, 2017)#[i for i in range(16 , 2017 )]                         # Artificial Meta-material dataset
FORCE_RUN = True
MODEL_NAME = None
#DATA_DIR = '/home/sr365/MM_Bench/Data/'                                               # All simulated simple dataset
#DATA_DIR = '/work/sr365/'                                      # real Meta-material dataset
#DATA_DIR = '/work/sr365/NN_based_MM_data/'                      # Artificial Meta-material dataset
#DATA_DIR = '/home/omar/PycharmProjects/github/idlm_Pytorch-master/forward/'
DATA_DIR = '../Data'
GEOBOUNDARY =[0.3, 0.6, 1, 1.5, 0.1, 0.2, -0.786, 0.786]
NORMALIZE_INPUT = True

# Running specific
USE_CPU_ONLY = False
EVAL_MODEL = "ga"

# NA Specific Parameters
USE_LORENTZ = False
LINEAR = [3, 400, 400, 400, 201]

CONV_OUT_CHANNEL = [4, 4, 4]
CONV_KERNEL_SIZE = [3, 3, 5]
CONV_STRIDE = [1, 1, 1]

# Optimizer Params
OPTIM = "Adam"
REG_SCALE = 5e-3
BATCH_SIZE = 1024
TRAIN_STEP = 300
LEARN_RATE = 1e-3
LR_DECAY_RATE = 0.8
