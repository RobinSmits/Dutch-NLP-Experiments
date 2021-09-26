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

# Uncomment .. For TF Debugging
# tf.config.run_functions_eagerly(True)

# Constants
MAX_LEN = 512 # Use the maximum input length for MT5 or ByT5
FOLD_SPLITS = 5
EPOCHS = 10
VERBOSE = 1
CACHE_DIR = './'
WORK_DIR = './'
SEEDS = [*range(1000, 1003, 1)]
FOLD_EARLY_STOP = 5

# Set Autotune
AUTO = tf.data.experimental.AUTOTUNE

# Set Batch Size
BASE_BATCH_SIZE = 32        # Modify to match your GPU card.
if tpu is not None:         
    BASE_BATCH_SIZE = 4     # TPU v3 ... (I've only had limited access to these through my Kaggle account..that gives you 30 hours for free each week..)
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

# Perform tokenization and labelling
input_ids, input_masks, output_ids, output_masks, labels = tokenize_t5_dpgnews_df(dpgnews_df, MAX_LEN, MAX_LABEL_LEN, tokenizer)

# Summary
print(f'\nInput Ids Shape: {input_ids.shape}')
print(f'Input Masks Shape: {input_masks.shape}')
print(f'Output Ids Shape: {output_ids.shape}')
print(f'Output Masks Shape: {output_masks.shape}')
print(f'Labels Shape: {labels.shape}')

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
    for fold, (train_index, val_index) in enumerate(folds.split(input_ids, labels)):
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
        
        # 'Cut Indexes' to contain only full size batches...
        # I had a lot of issues with both Loss and Accuracy occasionally going to NaN on TPU but not on GPU.
        # After finding this on github: https://github.com/tensorflow/tensorflow/issues/41635 ... 
        # I implemented some simple lines to make sure only full size batches are used during training and validation...that solved the NaN's
        train_count = len(train_index) // BATCH_SIZE
        val_count = len(val_index) // BATCH_SIZE
        train_index = train_index[:(train_count * BATCH_SIZE)]
        val_index = val_index[:(val_count * BATCH_SIZE)]
        print(train_index[:10])
        print(val_index[:10])

        # Create Train and Validation Array sets
        train_input_ids, train_input_masks, train_output_ids, train_output_masks, train_labels = input_ids[train_index], input_masks[train_index], output_ids[train_index], output_masks[train_index], labels[train_index]
        val_input_ids, val_input_masks, val_output_ids, val_output_masks, val_labels = input_ids[val_index], input_masks[val_index], output_ids[val_index], output_masks[val_index], labels[val_index]

        # Show Sizes
        print(f'Train Shape: {train_input_ids.shape}')
        print(f'Validation Shape: {val_input_ids.shape}')
        
        # Steps
        train_steps = train_input_ids.shape[0] // BATCH_SIZE
        val_steps = val_input_ids.shape[0] // BATCH_SIZE
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

        # Set Input
        train_input_data = {'input_ids': train_input_ids, 'labels': train_output_ids, 'attention_mask': train_input_masks, 'decoder_attention_mask': train_output_masks}
        validation_input_data = {'input_ids': val_input_ids, 'labels': val_output_ids, 'attention_mask': val_input_masks, 'decoder_attention_mask': val_output_masks}
               
        # Fit Model
        history = model.fit(train_input_data,
                            batch_size = BATCH_SIZE,
                            validation_data = validation_input_data,
                            epochs = EPOCHS, 
                            shuffle = True,
                            verbose = VERBOSE,
                            callbacks = [ModelCheckpoint(f'{WORK_DIR}model_fold{fold}.h5')],
                            use_multiprocessing = False,
                            workers = 4)
        
        # Validation Information
        best_val_accuracy = max(history.history['val_accuracy'])
        val_acc_list.append(best_val_accuracy)
        print(f'\n================================ Detection Accuracy: {best_val_accuracy * 100}%\n')

# Summary
print(f'Final Mean Accuracy for {FOLD_SPLITS} Fold CV Training: {np.mean(val_acc_list)}')