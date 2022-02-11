# Import Modules
import gc
import pandas as pd
import numpy as np 
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
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
except:
    strategy = tf.distribute.get_strategy()

# Uncomment .. For TF Debugging
# tf.config.run_functions_eagerly(True)

# Constants
MAX_LEN = 512
FOLD_SPLITS = 5
EPOCHS = 3
LR = 0.00002
VERBOSE = 1
WORK_DIR = './'
CACHE_DIR = './'
SEEDS = [*range(1000, 1003, 1)]

################## MODEL SETTINGS ###########################################################################
# Set Model Type for Base Model to use
    # 1. 'bert-base-multilingual-cased'    for Multilingual BERT model
    # 2. 'xlm-roberta-base'                for Multi-lingual XLM-RoBERTa model
model_type = 'bert-base-multilingual-cased'
# Set Model Architecture (Standard or Custom Sequence Classifier) 
    # True = Standard Sequence Classifier
    # False = Custom Sequence Classifier
use_standard_model = True
# Set Model Weights Type (Use Standard or Custom Pretrained on DPGNews Dataset) 
    # True = Default Model Weights as available from Huggingface Transformers
    # False = Custom Pretrained MLM model weights (pretrained with script: mlm_pretrain_mbert_xlmroberta.py)
use_default_weights = True
# Set Custom Pretrained Model Checkpoint Path (Used if 'use_default_model_weights = True)
custom_pretrained_model_checkpoint = './mlm_pretrain/bert-base-multilingual-cased/checkpoint-11000'

# Model Summary
print(f'Model Type: {model_type}')
print(f'Use Standard Model: {use_standard_model}')
print(f'Use Default Model Weights: {use_default_weights}')
print(f'Custom Pretrained Model Checkpoint: {custom_pretrained_model_checkpoint}')

# Set Batch Size
BASE_BATCH_SIZE = 4         # Modify to match your GPU card.
if tpu is not None:         
    BASE_BATCH_SIZE = 8     # TPU v2 or up...
BATCH_SIZE = BASE_BATCH_SIZE * strategy.num_replicas_in_sync

# Summary
print(f'Seeds: {SEEDS}')
print(f'Replica Count: {strategy.num_replicas_in_sync}')
print(f'Batch Size: {BATCH_SIZE}')
print(f'Learning Rate: {LR}')

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

        # Create Model
        if model_type == 'xlm-roberta-base' and use_standard_model == True:
            model = create_xlm_roberta_model_v1(use_default_weights, custom_pretrained_model_checkpoint, strategy, config, LR)
        if model_type == 'xlm-roberta-base' and use_standard_model == False:
            model = create_xlm_roberta_model_v2(use_default_weights, custom_pretrained_model_checkpoint, strategy, config, MAX_LEN, LR)
        if model_type == 'bert-base-multilingual-cased' and use_standard_model == True:
            model = create_mbert_model_v1(use_default_weights, custom_pretrained_model_checkpoint, strategy, config, LR)
        if model_type == 'bert-base-multilingual-cased' and use_standard_model == False:
            model = create_mbert_model_v2(use_default_weights, custom_pretrained_model_checkpoint, strategy, config, MAX_LEN, LR)

        # Model Summary
        if fold == 0: # Only need to show Model Summary once...
            model.summary()

        # Create Train and Validation Datasets
        train_dataset = create_dataset(train_df, MAX_LEN, tokenizer, BATCH_SIZE, shuffle = True)
        validation_dataset = create_dataset(val_df, MAX_LEN, tokenizer, BATCH_SIZE, shuffle = False)

        # Steps
        train_steps = train_df.shape[0] // BATCH_SIZE
        val_steps = val_df.shape[0] // BATCH_SIZE
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

        # Classification Report
        preds = model.predict(validation_dataset, verbose = VERBOSE)
        print(f'\n================================ Classification Report')
        if use_standard_model:
            print(classification_report(np.hstack([label.numpy() for example, label in validation_dataset]), preds.logits.argmax(axis = 1)))
        else:
            print(classification_report(np.hstack([label.numpy() for example, label in validation_dataset]), preds.argmax(axis = 1)))
            
        # Cleanup
        del train_df, val_df, train_dataset, validation_dataset, model
        gc.collect()

# Summary
print(f'Final Mean Accuracy for Multiple Seeds / Fold CV Training: {np.mean(val_acc_list)}')