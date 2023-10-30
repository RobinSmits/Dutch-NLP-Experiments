# Import Modules
from sklearn.model_selection import train_test_split
from transformers import (AutoConfig,
                          AutoTokenizer,
                          AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling,
                          TrainingArguments,
                          Trainer)

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

# Set Model Type for Base Model to use
    # 1. 'bert-base-multilingual-cased'        for Multi-lingual BERT model
    # 2. 'distilbert-base-multilingual-cased'  for Multi-lingual DistilBert model
    # 3. 'xlm-roberta-base'                    for Multi-lingual XLM-RoBERTa model
model_type = 'distilbert-base-multilingual-cased'
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
# As explained in the Readme.md the required files are only available on: https://www.kaggle.com/datasets/rsmits/dpgmedia2019
# Make sure they are located in: CACHE_DIR
# download_articles_by_publisher(CACHE_DIR)
# Get DpgNews Dataframe
dpgnews_df = get_dpgnews_df(CACHE_DIR)

# Split into Train and Validation
train_df, val_df = train_test_split(dpgnews_df, test_size = TEST_SIZE, random_state = SEED, shuffle = True)

# Create Data Collator
datacollator = DataCollatorForLanguageModeling(tokenizer = tokenizer, 
                                               mlm = True, 
                                               mlm_probability = 0.15)

# Create dataset
ds = create_dataset_for_pretraining(tokenizer, train_df, val_df)

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
                                  save_total_limit = 3,
                                  save_steps = 1000,
                                  eval_steps = 1000,
                                  metric_for_best_model = 'eval_loss',
                                  greater_is_better = False,
                                  load_best_model_at_end = True,
                                  prediction_loss_only = True)

# Set Trainer
trainer = Trainer(model = model,
                  args = training_args,
                  data_collator = datacollator,
                  train_dataset = ds['train'],
                  eval_dataset = ds['validation'])

# Train MLM Model
trainer.train()