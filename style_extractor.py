from __future__ import division
from __future__ import print_function
import tensorflow as tf
import collections
import numpy as np
import os
import cPickle
import configurations
from skipthoughts import encoder_manager
np.random.seed(99)

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("stv_vocab", "", "path to vocab used by stv (expanded)")
tf.flags.DEFINE_string("stv_embedding", "", "path stv embeddings of vocabulary")
tf.flags.DEFINE_string("stv_model", "", "path of pretrained skipthought model checkpoint")
tf.flags.DEFINE_string("book_data_dir", "", "path to directory of book data")
tf.flags.DEFINE_string("book_category", "Adventure", "cateogry of book")

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
os.environ["CUDA_VISIBLE_DEVICES"]="7"
tf.logging.set_verbosity(tf.logging.INFO)

def load_book_vocab(book_file, most_common, num_sample_passage):

    tf.logging.info("Reading original book from %s", book_file)
    tf.logging.info("Sampling %s and take most common %s vocabulary", num_sample_passage, most_common)

    book_data = []
    for f in tf.gfile.Glob(FLAGS.book_data_dir + FLAGS.book_category + "/*"):
        with tf.gfile.Open(f) as book:
            for line in book:
                passage = line.decode("utf-8").strip()
                book_data.append(passage)
    
    np.random.shuffle(book_data)
    book_sample_passage = book_data[0:num_sample_passage]
    
    return book_sample_passage      

def main(unused_arg):
    book_sample_passage = load_book_vocab(FLAGS.book_data_dir + FLAGS.book_category + "/*",\
                                          num_sample_passage=-1)
    style_length_cut_long = 100 #>50(adventure:50), (romance:100) word for verbose stylem <50 for terse one
    style_length_cut_short = 50 
    sentence_long = []
    sentence_short = []
    ad_hoc_cut = 610
    for p in book_sample_passage:
        if len(p) > ad_hoc_cut:
            continue
        elif len(p.split(" ")) >= style_length_cut_long:
            sentence_long.append(p)
        
        elif len(p.split(" ")) <= style_length_cut_short:
            sentence_short.append(p)

    min_len_long = min([len(e.split(" ")) for e in sentence_long])
    max_len_short = max([len(e.split(" ")) for e in sentence_short])
    print(min_len_long)
    print(max_len_short)

    assert min_len_long >= style_length_cut_long and max_len_short <= style_length_cut_short

    tf.logging.info("Number of passage long: %s", len(sentence_long))
    tf.logging.info("Number of passage short: %s", len(sentence_short))

    tf.logging.info("loading encoder")
    stv_config = stv_config()
    encoder = encoder_manager.EncoderManager(config_hardware=config)
    encoder.load_model(model_config=stv_config,
                       vocabulary_file=FLAGS.stv_vocab,
                       embedding_matrix_file=FLAGS.stv_embedding,
                       checkpoint_path=FLAGS.stv_model)
    tf.logging.info("encoding long sentence")
    encodings_long = encoder.encode(sentence_long)
    tf.logging.info("encoding short sentence")
    encodings_short = encoder.encode(sentence_short)

    encodings_valid_long = []
    encodings_valid_short =[]
    num_invalid_encoding = 0

    for e in encodings_long:        
        if np.isnan(e).any():
            num_invalid_encoding+=1
        else:
            encodings_valid_long.append(e)

    for e in encodings_short:
        if np.isnan(e).any():
            num_invalid_encoding+=1
        else:
            encodings_valid_short.append(e)

    tf.logging.info("Number of total invalid encodings: %s", num_invalid_encoding)
    assert (len(encodings_valid_long) + len(encodings_valid_short) + num_invalid_encoding) == (len(encodings_long) + len(encodings_short))

    style_bias_long = np.mean(encodings_valid_long, axis=0)
    style_bias_short = np.mean(encodings_valid_short, axis=0)

    with open("style_bias/bias_fulladvent_long_100.pkl", 'w') as handle:
        cPickle.dump(style_bias_long, handle)
    with open("style_bias/bias_fulladvent_short_50.pkl", 'w') as handle:
        cPickle.dump(style_bias_short, handle)

if __name__ == "__main__":
    tf.app.run()

