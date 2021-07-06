import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from urllib.request import urlopen

def download_articles_by_publisher(cache_dir: str):
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
    dpgnews_df = dpgnews_df.sample(frac = 1.0)

    return dpgnews_df

def tokenize_dpgnews_df(df: pd.DataFrame, max_len: int, tokenizer):
    total_samples = df.shape[0]

    # Placeholders input
    input_ids = np.zeros((total_samples, max_len), dtype = 'int32')
    input_masks = np.zeros((total_samples, max_len), dtype = 'int32')
    labels = np.zeros((total_samples, ), dtype = 'int32')

    for index, row in tqdm(zip(range(0, total_samples), df.iterrows()), total = total_samples):
        
        # Get title and description as strings
        text = row[1]['text']
        partisan = row[1]['partisan']

        # Process Description - Set Label for real as 0
        input_encoded = tokenizer.encode_plus(text, add_special_tokens = True, max_length = max_len, truncation = True, padding = 'max_length')
        input_ids_sample = input_encoded['input_ids']
        input_ids[index,:] = input_ids_sample
        attention_mask_sample = input_encoded['attention_mask']
        input_masks[index,:] = attention_mask_sample
        labels[index] = 1 if partisan == 'true' else 0

    # Return Arrays
    return input_ids, input_masks, labels
   
def create_dataset(input_ids: np.array, input_masks: np.array, labels: np.array):
    # Create DatasetDictionary structure is also preserved.
    dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': input_masks}, labels))

    # Return Dataset
    return dataset

def create_train_dataset(input_ids: np.array, input_masks: np.array, labels: np.array, batch_size: int):
    train_dataset = create_dataset(input_ids, input_masks, labels)
    train_dataset = train_dataset.shuffle(1024, reshuffle_each_iteration = True)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.repeat(-1)
    train_dataset = train_dataset.prefetch(1024)

    return train_dataset

def create_validation_dataset(input_ids: np.array, input_masks: np.array, labels: np.array, batch_size: int):
    validation_dataset = create_dataset(input_ids, input_masks, labels)
    validation_dataset = validation_dataset.batch(batch_size)
    validation_dataset = validation_dataset.repeat(-1)
    validation_dataset = validation_dataset.prefetch(1024)

    return validation_dataset