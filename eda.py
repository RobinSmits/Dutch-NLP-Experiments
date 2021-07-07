# Import Modules
import pandas as pd
import numpy as np 
from tqdm import tqdm
from transformers import *
import matplotlib.pyplot as plt
import seaborn as sns

# Custom Code
from dataset import *

# Constants
MAX_LEN = 512
CACHE_DIR = './'

# Set Model Type
# Set to the following:
# 1. 'bert-base-multilingual-cased'    for Multilingual BERT model
# 2. 'xlm-roberta-base'                for Multi-lingual XLM-RoBERTa model
model_type = 'xlm-roberta-base'
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

# Get Counts for Title, Text, Sentences and Partisan Labels
dpgnews_df['title_wordcount'] = dpgnews_df.title.str.split(' ').str.len()
dpgnews_df['text_wordcount'] = dpgnews_df.text.str.split(' ').str.len()
dpgnews_df['text_sentence_count'] = dpgnews_df.text.str.split('.').str.len()
print('\n===== Partisan - Label Count')
print(dpgnews_df.partisan.value_counts().sort_index())
print('\n===== Title - Word Count')
print(dpgnews_df.title_wordcount.value_counts().sort_index())
print('\n===== Text - Word Count')
print(dpgnews_df.text_wordcount.value_counts().sort_index())
print('\n===== Text - Sentence Count')
print(dpgnews_df.text_sentence_count.value_counts().sort_index())
    
# Plot Words Count
g = sns.displot(dpgnews_df, kind = 'kde', rug = True, x = 'text_wordcount', hue = 'partisan')
g.set_axis_labels("Text - Words Per Article Count", 'Density', labelpad = 10)
g.fig.set_size_inches(10, 6)
plt.savefig(f'{model_type}_plot_text_words_count.png', dpi = 200)
plt.close()

# Plot Sentence Count
g = sns.displot(dpgnews_df, kind = 'kde', rug = True, x = 'text_sentence_count', hue = 'partisan')
g.set_axis_labels("Text - Sentences Per Article Count", 'Density', labelpad = 10)
g.fig.set_size_inches(10, 6)
plt.savefig(f'{model_type}_plot_text_sentences_count.png', dpi = 200)
plt.close()

# Get Token Count / PLot Token Count
dpgnews_df['text_token_count'] = 0
for index, row in tqdm(dpgnews_df.iterrows(), total = dpgnews_df.shape[0]):
    # Get title and description as strings
    text = row['text']
    
    # Get the full tokenized Text... No Max Length, No Truncation etc...
    input_encoded = tokenizer.encode_plus(text, add_special_tokens = True)
    dpgnews_df.loc[index, 'text_token_count'] = len(input_encoded['input_ids']) 
g = sns.displot(dpgnews_df, kind = 'kde', rug = True, x = 'text_token_count', hue = 'partisan')
g.set_axis_labels(f'Text {model_type} Tokens Per Article Count', 'Density', labelpad = 10)
g.fig.set_size_inches(10, 6)
plt.savefig(f'{model_type}_plot_text_token_count.png', dpi = 200)
plt.close()

# Text Token Counts Smaller/Greater than 512 .. which is max input size for Transformers model
print(f'\nArticles with 512 or less tokens: {dpgnews_df[dpgnews_df["text_token_count"] <= 512].shape[0]}')
print(f'Articles with more than 512 tokens: {dpgnews_df[dpgnews_df["text_token_count"] > 512].shape[0]}')