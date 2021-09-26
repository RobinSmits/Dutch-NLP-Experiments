import tensorflow as tf
import tensorflow_addons as tfa
from transformers import *

def ModelCheckpoint(model_name: str)->tf.keras.callbacks.ModelCheckpoint:
    return tf.keras.callbacks.ModelCheckpoint(model_name, 
                                              monitor = 'val_accuracy', 
                                              verbose = 1, 
                                              save_best_only = True, 
                                              save_weights_only = True, 
                                              mode = 'max', 
                                              period = 1)

### XLM-RoBERTa ######################################################################################################

def create_xlm_roberta_model_v1(use_default_weights: bool, 
                                custom_pretrained_model_checkpoint: str, 
                                strategy: tf.distribute.Strategy, 
                                config: AutoConfig, 
                                lr: float)->tf.keras.Model:
    
    # Set Model init specs
    if use_default_weights:
        model_type = 'jplu/tf-xlm-roberta-base'
        from_pt = False
    else:
        model_type = custom_pretrained_model_checkpoint
        from_pt = True
        
    # Create 'Standard' Classification Model
    with strategy.scope():   
        model = TFXLMRobertaForSequenceClassification.from_pretrained(model_type, config = config, from_pt = from_pt)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

        model.compile(optimizer = optimizer, loss = loss, metrics = [metric])        
        
        return model

def create_xlm_roberta_model_v2(use_default_weights: bool, 
                                custom_pretrained_model_checkpoint: str, 
                                strategy: tf.distribute.Strategy, 
                                config: AutoConfig, 
                                max_len: int, 
                                lr: float)->tf.keras.Model:

    # Set Model init specs
    if use_default_weights:
        model_type = 'jplu/tf-xlm-roberta-base'
        from_pt = False
    else:
        model_type = custom_pretrained_model_checkpoint
        from_pt = True
    
    # Create Custom Model
    with strategy.scope():   
        input_ids = tf.keras.layers.Input(shape = (max_len,), dtype = tf.int32, name = 'input_ids')
        input_masks = tf.keras.layers.Input(shape = (max_len,), dtype = tf.int32, name = 'attention_mask')
        
        # Initializers
        kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05, seed = None)
        bias_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05, seed = None)

        transformers_model = TFRobertaModel.from_pretrained(model_type, config = config, from_pt = from_pt)
        
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

def create_xlm_roberta_model_v3(model_type: str, 
                                strategy: tf.distribute.Strategy, 
                                config: AutoConfig, 
                                max_len: int)->tf.keras.Model:
    
    # Create Custom Model
    with strategy.scope():   
        input_ids = tf.keras.layers.Input(shape = (max_len,), dtype = tf.int32, name = 'input_ids')
        input_masks = tf.keras.layers.Input(shape = (max_len,), dtype = tf.int32, name = 'attention_mask')
        
        transformers_model = TFRobertaModel.from_pretrained('jplu/tf-xlm-roberta-base', config = config)
        
        output_dict = transformers_model({'input_ids': input_ids, 'attention_mask': input_masks})
        last_hidden_state = output_dict.last_hidden_state
        outputs = last_hidden_state[:, 0, :]
        model = tf.keras.Model(inputs = [input_ids, input_masks], outputs = outputs) 
        
    return model

### Multi-Lingual BERT ######################################################################################################

def create_mbert_model_v1(use_default_weights: bool, 
                          custom_pretrained_model_checkpoint: str,
                          strategy: tf.distribute.Strategy, 
                          config: AutoConfig, 
                          lr: float)->tf.keras.Model:

    # Set Model init specs
    if use_default_weights:
        model_type = 'bert-base-multilingual-cased'
        from_pt = False
    else:
        model_type = custom_pretrained_model_checkpoint
        from_pt = True
    
    # Create 'Standard' Classification Model
    with strategy.scope():   
        model = TFBertForSequenceClassification.from_pretrained(model_type, config = config, from_pt = from_pt)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

        model.compile(optimizer = optimizer, loss = loss, metrics = [metric])        
        
        return model

def create_mbert_model_v2(use_default_weights: bool, 
                          custom_pretrained_model_checkpoint: str, 
                          strategy: tf.distribute.Strategy, 
                          config: AutoConfig, 
                          max_len: int, 
                          lr: float)->tf.keras.Model:

    # Set Model init specs
    if use_default_weights:
        model_type = 'bert-base-multilingual-cased'
        from_pt = False
    else:
        model_type = custom_pretrained_model_checkpoint
        from_pt = True
    
    # Create Custom Model
    with strategy.scope():   
        input_ids = tf.keras.layers.Input(shape = (max_len,), dtype = tf.int32, name = 'input_ids')
        input_masks = tf.keras.layers.Input(shape = (max_len,), dtype = tf.int32, name = 'attention_mask')
        
        # Initializers
        kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05, seed = None)
        bias_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05, seed = None)

        transformers_model = TFBertModel.from_pretrained(model_type, config = config, from_pt = from_pt)
        
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

def create_mbert_model_v3(model_type: str, 
                          strategy: tf.distribute.Strategy, 
                          config: AutoConfig, 
                          max_len: int)->tf.keras.Model:

    # Create Custom Model
    with strategy.scope():   
        input_ids = tf.keras.layers.Input(shape = (max_len,), dtype = tf.int32, name = 'input_ids')
        input_masks = tf.keras.layers.Input(shape = (max_len,), dtype = tf.int32, name = 'attention_mask')
        
        transformers_model = TFBertModel.from_pretrained(model_type, config = config)
        
        output_dict = transformers_model({'input_ids': input_ids, 'attention_mask': input_masks})
        last_hidden_state = output_dict.last_hidden_state
        outputs = last_hidden_state[:, 0, :]
        model = tf.keras.Model(inputs = [input_ids, input_masks], outputs = outputs) 
        
        return model

### MT5 and ByT5 ############################################################################################################

class T5_Accuracy(tf.keras.metrics.Metric):
    def __init__(self, label_length, name = 'accuracy', **kwargs):
        super(T5_Accuracy, self).__init__(name = name, **kwargs)
        self.t5_accuracy = self.add_weight(name = 'accuracy', initializer = 'zeros')
        self.steps_counter = self.add_weight(name = 'steps_counter', initializer = 'zeros')
        self.label_length = label_length

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight = None):
        # Reshape
        y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        y_true = tf.reshape(y_true, [-1])

        # Get Max Indexes
        y_pred = tf.math.argmax(y_pred, 1, output_type = 'int32')

        # Cast to Int32
        y_true = tf.cast(y_true, 'int32')
        
        # Reshape according to max label length...we want to compare the exact predictions made.
        y_pred = tf.reshape(y_pred, [-1, self.label_length])
        y_true = tf.reshape(y_true, [-1, self.label_length])
        
        # Compare Predicted and Labelled
        y_comparison = tf.math.equal(y_pred, y_true)
        
        accuracy = tf.keras.backend.mean(tf.cast(tf.math.reduce_all(y_comparison, 1), tf.keras.backend.floatx()))
        self.t5_accuracy.assign_add(accuracy)
        self.steps_counter.assign_add(tf.ones(shape = ()))
        
    @tf.function    
    def result(self):
        return self.t5_accuracy / self.steps_counter

    @tf.function    
    def reset_state(self):
        for var in self.variables:
            var.assign(tf.zeros(shape = var.shape))

class KerasTFMT5ForConditionalGeneration(TFMT5ForConditionalGeneration):
    def __init__(self, *args, log_dir=None, cache_dir= None, **kwargs):
        super().__init__(*args, **kwargs)

        self.loss_tracker= tf.keras.metrics.Mean(name = 'loss') 
    
    @tf.function
    def train_step(self, data):
        x = data[0]
        y = x['labels']
        y = tf.reshape(y, [-1, 1])
        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            loss = outputs[0]
            logits = outputs[1]
            loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, self.trainable_variables)
            
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)        
        self.compiled_metrics.update_state(y, logits)
        metrics = {m.name: m.result() for m in self.metrics}
        
        return metrics

    def test_step(self, data):
        x = data[0]
        y = x["labels"]
        y = tf.reshape(y, [-1, 1])
        output = self(x, training = False)
        loss = output[0]
        loss = tf.reduce_mean(loss)
        logits = output[1]
        
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, logits)
        
        return {m.name: m.result() for m in self.metrics}

class KerasTFByT5ForConditionalGeneration(TFT5ForConditionalGeneration):
    def __init__(self, *args, log_dir=None, cache_dir= None, **kwargs):
        super().__init__(*args, **kwargs)

        self.loss_tracker= tf.keras.metrics.Mean(name = 'loss') 
    
    @tf.function
    def train_step(self, data):
        x = data[0]
        y = x['labels']
        y = tf.reshape(y, [-1, 1])
        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            loss = outputs[0]
            logits = outputs[1]
            loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, self.trainable_variables)
            
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)        
        self.compiled_metrics.update_state(y, logits)
        metrics = {m.name: m.result() for m in self.metrics}
        
        return metrics

    def test_step(self, data):
        x = data[0]
        y = x["labels"]
        y = tf.reshape(y, [-1, 1])
        output = self(x, training = False)
        loss = output[0]
        loss = tf.reduce_mean(loss)
        logits = output[1]
        
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, logits)
        
        return {m.name: m.result() for m in self.metrics}

def create_mt5_model(model_type: str, strategy: tf.distribute.Strategy, config: AutoConfig, lr: float, max_label_len: int, total_steps: int)->tf.keras.Model:
    # Create Model
    with strategy.scope():
        radam = tfa.optimizers.RectifiedAdam(lr = lr, total_steps = total_steps, warmup_proportion = 0.10, min_lr = lr/3.)
        ranger = tfa.optimizers.Lookahead(radam, sync_period = 6, slow_step_size = 0.5)

        model = KerasTFMT5ForConditionalGeneration.from_pretrained(model_type, config = config)
        model.compile(optimizer = ranger, metrics = [T5_Accuracy(label_length = max_label_len)])
        
        return model

def create_byt5_model(model_type: str, strategy: tf.distribute.Strategy, config: AutoConfig, lr: float, max_label_len: int, total_steps: int)->tf.keras.Model:
    # Create Model
    with strategy.scope():
        radam = tfa.optimizers.RectifiedAdam(lr = lr, total_steps = total_steps, warmup_proportion = 0.10, min_lr = lr/3.)
        ranger = tfa.optimizers.Lookahead(radam, sync_period = 6, slow_step_size = 0.5)

        model = KerasTFByT5ForConditionalGeneration.from_pretrained(model_type, config = config)
        model.compile(optimizer = ranger, metrics = [T5_Accuracy(label_length = max_label_len)])
        
        return model