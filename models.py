import tensorflow as tf
from transformers import *

def ModelCheckpoint(model_name):
    return tf.keras.callbacks.ModelCheckpoint(model_name, 
                                              monitor = 'val_accuracy', 
                                              verbose = 1, 
                                              save_best_only = True, 
                                              save_weights_only = True, 
                                              mode = 'max', 
                                              period = 1)

### XLM-RoBERTa ######################################################################################################

def create_xlm_roberta_model_v1(strategy, config, lr):
    # Create 'Standard' Classification Model
    with strategy.scope():   
        model = TFXLMRobertaForSequenceClassification.from_pretrained('jplu/tf-xlm-roberta-base', config = config)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

        model.compile(optimizer = optimizer, loss = loss, metrics = [metric])        
        
        return model

def create_xlm_roberta_model_v2(strategy, config, max_len, lr):
    # Create Custom Model
    with strategy.scope():   
        input_ids = tf.keras.layers.Input(shape = (max_len,), dtype = tf.int32, name = 'input_ids')
        input_masks = tf.keras.layers.Input(shape = (max_len,), dtype = tf.int32, name = 'attention_mask')
        
        # Initializers
        kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05, seed = None)
        bias_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05, seed = None)

        transformers_model = TFRobertaModel.from_pretrained('jplu/tf-xlm-roberta-base', config = config)
        
        last_hidden_states = transformers_model({'input_ids': input_ids, 'attention_mask': input_masks})
        x = last_hidden_states[0][:, 0, :]
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(2, kernel_initializer = kernel_initializer, bias_initializer = bias_initializer)(x)
        model = tf.keras.Model(inputs = [input_ids, input_masks], outputs = outputs) 

        optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

        # Compile
        model.compile(optimizer = optimizer, loss = loss, metrics = [metric])        
        
        return model

### Multi-Lingual BERT ######################################################################################################

def create_mbert_model_v1(model_type, strategy, config, lr):
    # Create 'Standard' Classification Model
    with strategy.scope():   
        model = TFBertForSequenceClassification.from_pretrained(model_type, config = config)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

        model.compile(optimizer = optimizer, loss = loss, metrics = [metric])        
        
        return model

def create_mbert_model_v2(model_type, strategy, config, max_len, lr):
    # Create Custom Model
    with strategy.scope():   
        input_ids = tf.keras.layers.Input(shape = (max_len,), dtype = tf.int32, name = 'input_ids')
        input_masks = tf.keras.layers.Input(shape = (max_len,), dtype = tf.int32, name = 'attention_mask')
        
        # Initializers
        kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05, seed = None)
        bias_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05, seed = None)

        transformers_model = TFBertModel.from_pretrained(model_type, config = config)
        
        last_hidden_states = transformers_model({'input_ids': input_ids, 'attention_mask': input_masks})
        x = last_hidden_states[0][:, 0, :]
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(2, kernel_initializer = kernel_initializer, bias_initializer = bias_initializer)(x)
        model = tf.keras.Model(inputs = [input_ids, input_masks], outputs = outputs) 

        optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

        # Compile
        model.compile(optimizer = optimizer, loss = loss, metrics = [metric])        
        
        return model