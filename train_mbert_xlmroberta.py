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

# Configure Strategy. Assume TPU...if not set default for GPU/CPU
tpu = None
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    #tf.config.set_visible_devices([], 'GPU') # Uncomment to force tensorflow to use CPU instead of GPU
    strategy = tf.distribute.get_strategy()

# Constants
MAX_LEN = 512
FOLD_SPLITS = 5
EPOCHS = 3
VERBOSE = 1
WORK_DIR = './'
CACHE_DIR = './'
SEEDS = [*range(1000, 1003, 1)]

# Set Autotune
AUTO = tf.data.experimental.AUTOTUNE

# Set Batch Size
BASE_BATCH_SIZE = 3         # Modify to match your GPU card.
if tpu is not None:         
    BASE_BATCH_SIZE = 8     # TPU v2 or up...
BATCH_SIZE = BASE_BATCH_SIZE * strategy.num_replicas_in_sync

# Set Learning Rate
LR = 0.00002

# Summary
print(f'Seeds: {SEEDS}')
print(f'Replica Count: {strategy.num_replicas_in_sync}')
print(f'Batch Size: {BATCH_SIZE}')
print(f'Learning Rate: {LR}')

# Set Model Type
# Set to the following:
# 1. 'bert-base-multilingual-cased'    for Multilingual BERT model
# 2. 'xlm-roberta-base'                for Multi-lingual XLM-RoBERTa model
model_type = 'bert-base-multilingual-cased'
print(f'Model Type: {model_type}')

# Set Config
config = AutoConfig.from_pretrained(model_type, num_labels = 2) # Binary classification so set num_labels = 2
print(config)

# Set Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_type, add_prefix_space = False, do_lower_case = False)
print(tokenizer)

# Download DpgNews Files
download_articles_by_publisher(CACHE_DIR)

# Get DpgNews Dataframe
dpgnews_df = get_dpgnews_df(CACHE_DIR)

# Perform tokenization and labelling
input_ids, input_masks, labels = tokenize_dpgnews_df(dpgnews_df, MAX_LEN, tokenizer)

# Summary
print(f'\nInput Ids Shape: {input_ids.shape}')
print(f'Input Masks Shape: {input_masks.shape}')
print(f'Labels Shape: {labels.shape}')

# Accuracy PlaceHoler
val_acc_list = []

# Loop through SEEDS
for seed in SEEDS:    
    # Seeds
    set_seeds(seed)

    # Create Folds
    folds = StratifiedKFold(n_splits = FOLD_SPLITS, shuffle = True, random_state = seed)

    # Loop through Folds
    for fold, (train_index, val_index) in enumerate(folds.split(input_ids, labels)):
        # START        
        print(f'\n================================ FOLD {fold} === SEED {seed}')
        
        # Cleanup
        tf.keras.backend.clear_session()    
        if tpu is not None:
            tf.tpu.experimental.initialize_tpu_system(tpu)
        gc.collect()
        
        # Show Indexes
        print(train_index[:10])
        print(val_index[:10])

        # Create Train and Validation Array sets
        train_input_ids, train_input_masks, train_labels = input_ids[train_index], input_masks[train_index], labels[train_index]
        val_input_ids, val_input_masks, val_labels = input_ids[val_index], input_masks[val_index], labels[val_index]
 
        # Show Sizes
        print(f'Train Shape: {train_input_ids.shape}')
        print(f'Validation Shape: {val_input_ids.shape}')

        # Create Train Dataset
        train_dataset = create_train_dataset(train_input_ids, train_input_masks, train_labels, BATCH_SIZE)

        # Create Validation Dataset
        validation_dataset = create_validation_dataset(val_input_ids, val_input_masks, val_labels, BATCH_SIZE)

        # Create Model
        #if model_type == 'xlm-roberta-base': model = create_xlm_roberta_model_v1(strategy, config, LR)
        if model_type == 'xlm-roberta-base': model = create_xlm_roberta_model_v2(strategy, config, MAX_LEN, LR)
        #if model_type == 'bert-base-multilingual-cased': model = create_mbert_model_v1(model_type, strategy, config, LR)
        if model_type == 'bert-base-multilingual-cased': model = create_mbert_model_v2(model_type, strategy, config, MAX_LEN, LR)

        # Model Summary
        if fold == 0: # Only need to show Model Summary once...
            model.summary()

        # Steps
        train_steps = train_input_ids.shape[0] // BATCH_SIZE
        val_steps = val_input_ids.shape[0] // BATCH_SIZE
        total_steps = train_steps * EPOCHS
        print(f'Train Steps: {train_steps}')
        print(f'Val Steps: {val_steps}')
        print(f'Total Steps: {total_steps}')

        # Fit Model
        model.fit(train_dataset,
                steps_per_epoch = train_steps,
                validation_data = validation_dataset,
                validation_steps = val_steps,
                epochs = EPOCHS, 
                verbose = VERBOSE,
                callbacks = [ModelCheckpoint(f'{WORK_DIR}model.h5')])

        # Evaluate Dataset
        model.load_weights(f'{WORK_DIR}model.h5') # Reload the Best Model
        eval = model.evaluate(validation_dataset, steps = val_steps, verbose = VERBOSE)
        val_acc_list.append(eval[1])
        print(f'\n================================ Detection Accuracy: {eval[1] * 100}%\n')

# Summary
print(f'Final Mean Accuracy for Multiple Seeds / Fold CV Training: {np.mean(val_acc_list)}')