import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Tuple
from urllib.request import urlopen

def download_articles_by_publisher(cache_dir: str)->None:
    # URLs taken from: https://github.com/dpgmedia/partisan-news2019
    articles_by_publisher_url = 'https://partisan-news2019.s3-eu-west-1.amazonaws.com/dpgMedia2019-articles-bypublisher.jsonl'
    labels_by_publisher_url = 'https://github.com/dpgmedia/partisan-news2019/raw/master/dpgMedia2019-labels-bypublisher.jsonl'

    # Articles By Publisher
    if os.path.isfile(os.path.join(cache_dir, 'dpgMedia2019-articles-bypublisher.jsonl')):
        print ("Articles By Publisher File exist")
    else:
        # Download...
        print ('Downloading: Articles By Publisher File....')

        # Download File and save
        with urlopen(articles_by_publisher_url) as file_stream:
            file_data = file_stream.read()
 
            with open(os.path.join(cache_dir, 'dpgMedia2019-articles-bypublisher.jsonl'), 'wb') as f:
                f.write(file_data)
            
    # Labels By Publisher
    if os.path.isfile(os.path.join(cache_dir, 'dpgMedia2019-labels-bypublisher.jsonl')):
        print('Labels By Publisher File exist')
    else:
        # Download...
        print ('Downloading: Labels By Publisher File....')

        # Download File and save
        with urlopen(labels_by_publisher_url) as file_stream:
            file_data = file_stream.read()
 
            with open(os.path.join(cache_dir, 'dpgMedia2019-labels-bypublisher.jsonl'), 'wb') as f:
                f.write(file_data)

def get_dpgnews_df(cache_dir: str)->pd.DataFrame:
    # Set 1: Articles
    articles_df = pd.read_json(os.path.join(cache_dir, 'dpgMedia2019-articles-bypublisher.jsonl'), lines = True)
    articles_df = articles_df.set_index('id')
    print(articles_df.shape) 

    # Set 2: Labels
    labels_df = pd.read_json(os.path.join(cache_dir, 'dpgMedia2019-labels-bypublisher.jsonl'), lines = True)
    labels_df = labels_df.set_index('id')
    print(labels_df.shape) 

    # Finalize Full Data
    dpgnews_df = articles_df.join(labels_df, on = ['id'], how = 'inner')
    print(dpgnews_df.shape) 

    # Randomize all rows...
    dpgnews_df = dpgnews_df.sample(frac = 1.0, random_state = 42)

    return dpgnews_df

def create_dataset(df: pd.DataFrame, max_len: int, tokenizer: AutoTokenizer, batch_size: int, shuffle = False)->tf.data.Dataset:
    total_samples = df.shape[0]

    # Placeholders input
    input_ids, input_masks = [], []
    
    # Placeholder output
    labels = []

    # Tokenize
    for index, row in tqdm(zip(range(0, total_samples), df.iterrows()), total = total_samples):
        
        # Get title and description as strings
        text = row[1]['text']
        partisan = row[1]['partisan']

        # Encode
        input_encoded = tokenizer.encode_plus(text, add_special_tokens = True, max_length = max_len, truncation = True, padding = 'max_length')
        input_ids.append(input_encoded['input_ids'])
        input_masks.append(input_encoded['attention_mask'])
        labels.append(1 if partisan == 'true' else 0)

    # Prepare and Create TF Dataset.
    all_input_ids = tf.constant(input_ids)
    all_input_masks = tf.constant(input_masks)
    all_labels = tf.constant(labels)
    dataset =  tf.data.Dataset.from_tensor_slices(({'input_ids': all_input_ids, 'attention_mask': all_input_masks}, all_labels))
    if shuffle:
        dataset = dataset.shuffle(1024, reshuffle_each_iteration = True)
    dataset = dataset.batch(batch_size, drop_remainder = True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def create_t5_dataset(df: pd.DataFrame, max_len: int, max_label_len: int, tokenizer: AutoTokenizer, batch_size: int, shuffle = False)->tf.data.Dataset:
    total_samples = df.shape[0]

    # Placeholders input
    input_ids, input_masks = [], []
    
    # Placeholders output
    output_ids, output_masks, labels = [], [], []

    # Tokenize
    for index, row in tqdm(zip(range(0, total_samples), df.iterrows()), total = total_samples):
        
        # Get title and description as strings
        text = row[1]['text']
        partisan = row[1]['partisan']
        
        # Process Input
        input_encoded = tokenizer.encode_plus('classificeer: ' + text, add_special_tokens = True, max_length = max_len, truncation = True, padding = 'max_length')
        input_ids.append(input_encoded['input_ids'])
        input_masks.append(input_encoded['attention_mask'])

        # Process Output
        labels.append(1 if partisan == 'true' else 0)
        partisan_label = 'politiek' if partisan == 'true' else 'neutraal'
        output_encoded = tokenizer.encode_plus(partisan_label, add_special_tokens = True, max_length = max_label_len, truncation = True, padding = 'max_length')
        output_ids.append(output_encoded['input_ids'])
        output_masks.append(output_encoded['attention_mask'])

    # Prepare and Create TF Dataset.
    all_input_ids = tf.constant(input_ids)
    all_output_ids = tf.constant(output_ids)
    all_input_masks = tf.constant(input_masks)
    all_output_masks = tf.constant(output_masks)
    dataset =  tf.data.Dataset.from_tensor_slices(({'input_ids': all_input_ids, 
                                                    'labels': all_output_ids, 
                                                    'attention_mask': all_input_masks, 
                                                    'decoder_attention_mask': all_output_masks}))
    if shuffle:
        dataset = dataset.shuffle(1024, reshuffle_each_iteration = True)
    dataset = dataset.batch(batch_size, drop_remainder = True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset