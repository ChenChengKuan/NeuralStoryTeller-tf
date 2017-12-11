class _HParams(object):
  """Wrapper for configuration parameters."""
  pass
class vse_config(object):
    # Image-sentence embedding
    def __init__(self):
        self.vsemodel = 'caption_gen/coco_embedding.npz' #'caption_gen/coco_embeddings/coco_embedding.npz'
        self.vgg = '' #path to vgg19.pkl 
        self.captions = 'caption_gen/coco_train_caps.txt'#'caption_gen/coco_embeddings/coco_train_caps.txt'

class storyteller_config(object):
    #story teller configuration
    def __init__(self):
        self.filename = ""
        self.values_per_input_shard = 2300
        self.batch_size = 32  #default 32
        self.vocab_size = None # To be computed from vocabulary file outside
        self.w2v_embedding_size = 620
        self.sk_embedding_size = 2400
        self.initializer_scale = 0.08
        self.code_name = "text_stv"
        self.text_name = "text"
        self.num_preprocess_threads = 1
        #LSTM Related
        self.lstm_dropout_keep_prob = 0.7 #default 0.7
        # LSTM input and output dimensionality, respectively.
        self.lstm_embedding_size = 620
        self.num_lstm_units = 800

class storyteller_training_config(object):
    #story teller training configuration
    def __init__(self):
        """Sets the default training hyperparameters."""
        # Number of examples per epoch of training data.
        self.num_examples_per_epoch = 231932

        # Optimizer for training the model.
        self.optimizer = "SGD" #default "SGD"

        # Learning rate for the initial phase of training.
        self.initial_learning_rate = 2.0    # default 2.0
        self.learning_rate_decay_factor = 0.8
        self.num_epochs_per_decay = 4  #default 8

        # If not None, clip gradients to this value.
        self.clip_gradients = 5.0

        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = 2


def stv_config(input_file_pattern=None,
                 input_queue_capacity=640000,
                 num_input_reader_threads=1,
                 shuffle_input_data=True,
                 uniform_init_scale=0.1,
                 vocab_size=20000,
                 batch_size=128,
                 word_embedding_dim=620,
                 bidirectional_encoder=False,
                 encoder_dim=2400):
    config = _HParams()
    config.input_file_pattern = input_file_pattern
    config.input_queue_capacity = input_queue_capacity
    config.num_input_reader_threads = num_input_reader_threads
    config.shuffle_input_data = shuffle_input_data
    config.uniform_init_scale = uniform_init_scale
    config.vocab_size = vocab_size
    config.batch_size = batch_size
    config.word_embedding_dim = word_embedding_dim
    config.bidirectional_encoder = bidirectional_encoder
    config.encoder_dim = encoder_dim
    return config


def stv_training_config(learning_rate=0.0008,
                    learning_rate_decay_factor=0.5,
                    learning_rate_decay_steps=400000,
                    number_of_steps=500000,
                    clip_gradient_norm=5.0,
                    save_model_secs=600,
                    save_summaries_secs=600):
    
    if learning_rate_decay_factor and not learning_rate_decay_steps:
        raise ValueError("learning_rate_decay_factor requires learning_rate_decay_steps.")
    config = _HParams()
    config.learning_rate = learning_rate
    config.learning_rate_decay_factor = learning_rate_decay_factor
    config.learning_rate_decay_steps = learning_rate_decay_steps
    config.number_of_steps = number_of_steps
    config.clip_gradient_norm = clip_gradient_norm
    config.save_model_secs = save_model_secs
    config.save_summaries_secs = save_summaries_secs
    return config
