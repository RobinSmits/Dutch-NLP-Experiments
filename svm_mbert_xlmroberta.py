# Import Modules
import gc
import pandas as pd
import numpy as np 
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
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

# Constants
MAX_LEN = 512
FOLD_SPLITS = 5
VERBOSE = 1
WORK_DIR = './'
CACHE_DIR = './'
SEEDS = [*range(1000, 1003, 1)]

################## MODEL SETTINGS ###########################################################################
# Set Model Type for Base Model to use
    # 1. 'bert-base-multilingual-cased'    for Multilingual BERT model
    # 2. 'xlm-roberta-base'                for Multi-lingual XLM-RoBERTa model
model_type = 'xlm-roberta-base'

# Model Summary
print(f'Model Type: {model_type}')

# Set Batch Size
BASE_BATCH_SIZE = 8         # Modify to match your GPU card.
if tpu is not None:         
    BASE_BATCH_SIZE = 8     # TPU v2 or up...
BATCH_SIZE = BASE_BATCH_SIZE * strategy.num_replicas_in_sync

# Summary
print(f'Seeds: {SEEDS}')
print(f'Replica Count: {strategy.num_replicas_in_sync}')

# Set Config
config = AutoConfig.from_pretrained(model_type)
config.output_hidden_states = True
config.return_dict = True
print(config)

# Set Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_type, add_prefix_space = False, do_lower_case = False)
print(tokenizer)

# Download DpgNews Files
download_articles_by_publisher(CACHE_DIR)

# Get DpgNews Dataframe
dpgnews_df = get_dpgnews_df(CACHE_DIR)

# Use a Transformers model with the default weights to extract feature vectors for Support Vector Classifier
# I modify the model to use the last hidden states layer to generate the feature vector.
# Different hidden states layers ( or combinations of hidden states layers) can be used to generate feature vectors.
if model_type == 'bert-base-multilingual-cased': feature_extraction_model = create_mbert_model_v3(model_type, strategy, config, MAX_LEN)
if model_type == 'xlm-roberta-base': feature_extraction_model = create_xlm_roberta_model_v3(model_type, strategy, config, MAX_LEN)
feature_extraction_model.summary()

# Create SVC Dataset
svc_dataset = create_dataset(dpgnews_df, MAX_LEN, tokenizer, BATCH_SIZE, shuffle = False)

# Get Feature Vectors for SVC - a feature vector for each input record...
print('\nGenerating Feature Vectors...')
svc_features_set = feature_extraction_model.predict(svc_dataset, verbose = 1)

# Get Labels from SVC dataset
labels = np.hstack([label.numpy() for example, label in svc_dataset])

# Accuracy PlaceHoler
val_acc_list = []

# Loop through SEEDS
for seed in SEEDS:    
    # Seeds
    set_seeds(seed)

    # Create Folds
    folds = StratifiedKFold(n_splits = FOLD_SPLITS, shuffle = True, random_state = seed)

    # Loop through Folds
    for fold, (train_index, val_index) in enumerate(folds.split(svc_features_set, labels)):
        # START        
        print(f'\n================================ FOLD {fold} === SEED {seed}')
        
        # Cleanup
        gc.collect()
        
        # Show Indexes
        print(train_index[:10])
        print(val_index[:10])

        # Create Train and Validation Array sets
        train_features, train_labels = svc_features_set[train_index], labels[train_index]
        val_features, val_labels = svc_features_set[val_index], labels[val_index]
 
        # Show Sizes
        print(f'Train Shape: {train_features.shape}')
        print(f'Validation Shape: {val_features.shape}')

        # Create Support Vector Machine Classifier
        # LinearSVC is used instead of SVC because it scales and performs very well with respect to the size of the dataset.
        # Earliest experiments I did with SVC just took to long. Based on a small subset of data the accuracy achieved is almost exactly the same
        svm_model = LinearSVC(penalty = 'l2', 
                              loss = 'squared_hinge', 
                              dual = False, 
                              tol = 0.001,  
                              C = 5.0, 
                              multi_class = 'ovr', 
                              fit_intercept = True, 
                              intercept_scaling = 1, 
                              verbose = 0, 
                              max_iter = 10000)
        
        # Fit SVM Model
        print('\n================================ Fitting SVM Model\n')
        svm_model.fit(train_features, train_labels)
        
        # Accuracy....
        eval_score = svm_model.score(val_features, val_labels)
        val_acc_list.append(eval_score)
        print(f'\n================================ Detection Accuracy: {eval_score * 100}%\n')

        # Classification Report
        preds = svm_model.predict(val_features   )
        print(f'\n================================ Classification Report')
        print(classification_report(val_labels, preds))
            
        # Cleanup
        del train_features, val_features, train_labels, val_labels, svm_model
        gc.collect()
        
# Summary
print(f'Final Mean Accuracy for Multiple Seeds / Fold CV Training: {np.mean(val_acc_list) * 100}%\n')
 