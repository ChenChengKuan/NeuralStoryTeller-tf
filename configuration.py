
class vse_config(object):
    # Image-sentence embedding
    def __init__(self):
        self.vsemodel = 'caption_gen/coco_embeddings/coco_embedding.npz'
        self.vgg = '/media/VSlab3/kuanchen_arxiv/NeuralStoryTeller/vgg19.pkl'
        self.captions = 'caption_gen/coco_embeddings/coco_train_caps.txt'
        
class _HParams(object):
  """Wrapper for configuration parameters."""
  pass

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