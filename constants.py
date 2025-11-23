from enum import Enum

INPUT_SIZE = 784  # MNIST images are 28x28, so 28*28=784
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 50

class Regime(Enum):
    NORMAL = "NORMAL"
    WARPED = "WARPED"
    WARPED_JOINT = "WARPED_JOINT"
    
# REGIME = Regime.NORMAL
REGIME = Regime.WARPED

if REGIME == Regime.NORMAL:
    OUTPUT_SIZE = 10  # 10 classes (digits 0-9)
elif REGIME == Regime.WARPED:
    OUTPUT_SIZE = 3  # variable vector size