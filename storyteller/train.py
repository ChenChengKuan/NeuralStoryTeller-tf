from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
from configs.config_teller import *
from story_teller import StoryDecoder
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
os.environ["CUDA_VISIBLE_DEVICES"]="2"


FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("input_file_name", "/media/VSlab3/kuanchen_arxiv/vocab_40000_100_advent/stv_text_code.tfrecord",
                       "path of input TFrecords")
tf.flags.DEFINE_string("input_vocab", "/media/VSlab3/kuanchen_arxiv/vocab_40000_100_advent/vocab_shared.txt", "vocabulary used ")
tf.flags.DEFINE_string("pretrained_embedding", "/media/VSlab3/kuanchen_arxiv/vocab_40000_100_advent/embeddings_r.npy", "pretrained reduced embeddings")
tf.flags.DEFINE_string("train_dir", "./res_adv_3",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_integer("number_of_steps", 600000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_boolean("pretrained_w2v", True, "Whether to use pretrained embedding")
tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    assert FLAGS.input_file_name, "--input_file_name is required"
    assert FLAGS.train_dir, "--train_dir is required"

    model_config = ModelConfig()
    model_config.filename = FLAGS.input_file_name
    vocab_count = 0
    with tf.gfile.Open(FLAGS.input_vocab, 'r') as f:
        for line in f:
            vocab_count += 1
    model_config.vocab_size = vocab_count
    training_config = TrainingConfig()


    train_dir = FLAGS.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)

    tf.logging.info("Building training graph.")
    g = tf.Graph()
    with g.as_default():
        
        model = StoryDecoder(
            model_config, mode="train")
        model.build()

        learning_rate_decay_fn = None

        learning_rate = tf.constant(training_config.initial_learning_rate)
        if training_config.learning_rate_decay_factor > 0:
            num_batches_per_epoch = (training_config.num_examples_per_epoch /
                                     model_config.batch_size)
            decay_steps = int(num_batches_per_epoch *
                              training_config.num_epochs_per_decay)

            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                  learning_rate,
                  global_step,
                  decay_steps=decay_steps,
                  decay_rate=training_config.learning_rate_decay_factor,
                  staircase=True)

            learning_rate_decay_fn = _learning_rate_decay_fn

        train_op = tf.contrib.layers.optimize_loss(
            loss=model.total_loss,
            global_step=model.global_step,
            learning_rate=learning_rate,
            optimizer=training_config.optimizer,
            clip_gradients=training_config.clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn)
        
        if FLAGS.pretrained_w2v:
            with open(FLAGS.pretrained_embedding) as f:
                embed_map_initial_values = np.load(f)
            
            var_names_to_values = {"embed_map":embed_map_initial_values}
            init_assign_op, init_feed_dict = tf.contrib.slim.assign_from_values(var_names_to_values)
            
            def InitAssignFn(sess):
                sess.run(init_assign_op,init_feed_dict)

        saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)
        
        tf.contrib.slim.learning.train(
          train_op,
          train_dir,
          log_every_n_steps=FLAGS.log_every_n_steps,
          graph=g,
          global_step=model.global_step,
          number_of_steps=FLAGS.number_of_steps,
          init_fn=InitAssignFn,
          session_config=config,
          saver=saver)

if __name__ == "__main__":
    tf.app.run()
