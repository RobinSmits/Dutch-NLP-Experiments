import random
import numpy as np
import pandas as pd
import tensorflow as tf

# Seeds
def set_seeds(seed: int)->None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed) 