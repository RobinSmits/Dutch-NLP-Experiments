# Import Modules
import gc
import pandas as pd
import numpy as np 
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
from transformers import *

# Custom Code
from dataset import *
from models import *
from utils import *

# Configure Strategy. Assume TPU...if not set default for GPU
tpu = None
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy()

# Uncomment .. For TF Debugging
# tf.config.run_functions_eagerly(True)

# Constants
MAX_LEN = 512 # Use the maximum input length for MT5 or ByT5
FOLD_SPLITS = 5
EPOCHS = 10
LR = 0.00002
VERBOSE = 1
CACHE_DIR = './'
WORK_DIR = './'
SEEDS = [*range(1000, 1003, 1)]
FOLD_EARLY_STOP = 5

# Set Batch Size
BASE_BATCH_SIZE = 4        # Modify to match your GPU card.
if tpu is not None:         
    BASE_BATCH_SIZE = 4     # TPU v3 ... (I've only had limited access to these through my Kaggle account..that gives you 30 hours for free each week..)
BATCH_SIZE = BASE_BATCH_SIZE * strategy.num_replicas_in_sync

# Summary
print(f'Seeds: {SEEDS}')
print(f'Replica Count: {strategy.num_replicas_in_sync}')
print(f'Batch Size: {BATCH_SIZE}')
print(f'Learning Rate: {LR}')

# Set Model Type
# Set to the following:
# 1. 'google/mt5-small' OR 'google/mt5-base'     for MT5 model. I haven't attempted larger on Kaggle TPUv3.
# 2. 'google/byt5-small' OR 'google/byt5-base'   for ByT5 model. I haven't attempted larger on Kaggle TPUv3.
model_type = 'google/mt5-small'
print(f'Model Type: {model_type}')

# Set Model Label Length !! Make sure it matches the model type...
# For MT5 ==> 3 ... 'politiek' / 'neutraal' is tokenized to 3 tokens.
# For ByT5 ==> 9 ... 'politiek' / 'neutraal' is tokenized to 9 tokens.
MAX_LABEL_LEN = 3

# Set Config
config = AutoConfig.from_pretrained(model_type)
print(config)

# Set Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_type, return_dict = True)
print(tokenizer)

# Download DpgNews Files
download_articles_by_publisher(CACHE_DIR)

# Get DpgNews Dataframe
dpgnews_df = get_dpgnews_df(CACHE_DIR)

# Label Example - True
print(f'\nLabelling Example: True ==> politiek')
partisan_label = 'politiek'
output_encoded = tokenizer.encode_plus(partisan_label, add_special_tokens = True, max_length = MAX_LABEL_LEN, truncation = True, padding = 'max_length')
print(output_encoded['input_ids'])
print(output_encoded['attention_mask'])
print(tokenizer.decode(output_encoded['input_ids']))

# Label Example - False
print(f'\nLabelling Example: False ==> neutraal')
partisan_label = 'neutraal'
output_encoded = tokenizer.encode_plus(partisan_label, add_special_tokens = True, max_length = MAX_LABEL_LEN, truncation = True, padding = 'max_length')
print(output_encoded['input_ids'])
print(output_encoded['attention_mask'])
print(tokenizer.decode(output_encoded['input_ids']))

# Accuracy PlaceHoler
val_acc_list = []

# Loop through SEEDS
for seed in SEEDS:    
    # Seeds
    set_seeds(seed)

    # Create Folds
    folds = StratifiedKFold(n_splits = FOLD_SPLITS, shuffle = True, random_state = seed)

    # Loop through Folds
    for fold, (train_index, val_index) in enumerate(folds.split(dpgnews_df, dpgnews_df.partisan.values)):
        # START        
        print(f'\n================================ FOLD {fold} === SEED {seed}')
        
        # Fold Early Stopping...To limit time required for training
        if fold > FOLD_EARLY_STOP:
            break
            
        # Cleanup
        tf.keras.backend.clear_session()    
        if tpu is not None:
            tf.tpu.experimental.initialize_tpu_system(tpu)
        gc.collect()
 
        # Show Indexes
        print(train_index[:10])
        print(val_index[:10])
        train_df = dpgnews_df.iloc[train_index]
        val_df = dpgnews_df.iloc[val_index]

        # Create Train and Validation Datasets
        train_dataset = create_t5_dataset(train_df, MAX_LEN, MAX_LABEL_LEN, tokenizer, BATCH_SIZE, shuffle = True)
        validation_dataset = create_t5_dataset(val_df, MAX_LEN, MAX_LABEL_LEN, tokenizer, BATCH_SIZE, shuffle = False)

        # Steps
        train_steps = train_df.shape[0] // BATCH_SIZE
        val_steps = val_df.shape[0] // BATCH_SIZE
        total_steps = train_steps * EPOCHS
        print(f'Train Steps: {train_steps}')
        print(f'Val Steps: {val_steps}')
        print(f'Total Steps: {total_steps}')

        # Create Model
        if 'mt5' in model_type:
            model = create_mt5_model(model_type, strategy, config, LR, MAX_LABEL_LEN, total_steps)
        if 'byt5' in model_type:
            model = create_byt5_model(model_type, strategy, config, LR, MAX_LABEL_LEN, total_steps)
        
        # Model Summary
        if fold == 0: # Only need to show Model Summary once...
            model.summary()

        # Fit Model
        history = model.fit(train_dataset,
                            steps_per_epoch = train_steps,
                            validation_data = validation_dataset,
                            validation_steps = val_steps,
                            epochs = EPOCHS, 
                            verbose = VERBOSE,
                            callbacks = [ModelCheckpoint(f'{WORK_DIR}model_fold{fold}.h5')])
                    
        # Validation Information
        best_val_accuracy = max(history.history['val_accuracy'])
        val_acc_list.append(best_val_accuracy)
        print(f'\n================================ Detection Accuracy: {best_val_accuracy * 100}%\n')

# Summary
print(f'Final Mean Accuracy for {FOLD_SPLITS} Fold CV Training: {np.mean(val_acc_list)}')