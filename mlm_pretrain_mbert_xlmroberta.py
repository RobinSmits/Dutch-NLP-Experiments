# Import Modules
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import *

# Custom Code
from dataset import *

# Constants
MAX_LEN = 512
EPOCHS = 2
VERBOSE = 1
CACHE_DIR = './'
SEED = 1000
LR = 0.000015
TEST_SIZE = 0.10

# Set Model Type .. Set to the following:
# 1. 'bert-base-multilingual-cased'    for Multilingual BERT model
# 2. 'xlm-roberta-base'                for Multi-lingual XLM-RoBERTa model
model_type = 'xlm-roberta-base'
print(f'Model Type: {model_type}')

# Set Config
config = AutoConfig.from_pretrained(model_type)
print(config)

# Set Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_type, add_prefix_space = False, do_lower_case = False)
print(tokenizer)

# Set MLM Model
model = AutoModelForMaskedLM.from_pretrained(model_type)

# Download DpgNews Files
download_articles_by_publisher(CACHE_DIR)

# Get DpgNews Dataframe
dpgnews_df = get_dpgnews_df(CACHE_DIR)

# Split into Train and Validation
train_df, val_df = train_test_split(dpgnews_df, test_size = TEST_SIZE, random_state = SEED, shuffle = True)

# Save Train Text File
with open('train_text.txt', 'w', encoding = 'utf-8') as f:
    for line in train_df.text.values.tolist():
        f.write(line + '\n')
    
# Save Validation Text File
with open('val_text.txt', 'w', encoding = 'utf-8') as f:
    for line in val_df.text.values.tolist():
        f.write(line + '\n')

# Create Train Dataset
train_dataset = LineByLineTextDataset(tokenizer = tokenizer,
                                      file_path = 'train_text.txt',
                                      block_size = MAX_LEN)

# Create Validation Dataset
valid_dataset = LineByLineTextDataset(tokenizer = tokenizer,
                                      file_path = 'val_text.txt',
                                      block_size = MAX_LEN)

# Create Data Collator
datacollator = DataCollatorForLanguageModeling(tokenizer = tokenizer, 
                                               mlm = True, 
                                               mlm_probability = 0.15)

# Set Training Arguments
training_args = TrainingArguments(output_dir = f'./mlm_pretrain/{model_type}/',
                                  learning_rate = LR,
                                  warmup_ratio = 0.10,
                                  gradient_accumulation_steps = 8,
                                  overwrite_output_dir = True,
                                  num_train_epochs = EPOCHS,
                                  per_device_train_batch_size = 2,
                                  per_device_eval_batch_size = 2,
                                  evaluation_strategy = 'steps',
                                  save_total_limit = 1,
                                  eval_steps = 1000,
                                  metric_for_best_model = 'eval_loss',
                                  greater_is_better = False,
                                  load_best_model_at_end = True,
                                  prediction_loss_only = True)

# Set Trainer
trainer = Trainer(model = model,
                  args = training_args,
                  data_collator = datacollator,
                  train_dataset = train_dataset,
                  eval_dataset = valid_dataset)

# Train MLM Model
trainer.train()