from __future__ import division
from __future__ import print_function
import tensorflow as tf
import collections
import numpy as np
import os
import cPickle
from configuration import *
from skipthought import encoder_manager
np.random.seed(99)

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("stv_vocab", "/media/VSlab3/kuanchen_arxiv/skip_thoughts_bi_2017_02_16/vocab.txt", "path to vocab used by stv (expanded)")
tf.flags.DEFINE_string("stv_embedding", "/media/VSlab3/kuanchen_arxiv/skip_thoughts_bi_2017_02_16/embeddings.npy", "path stv embeddings of vocabulary")
tf.flags.DEFINE_string("stv_model", "/media/VSlab3/kuanchen_arxiv/skip_thoughts_bi_2017_02_16/model.ckpt-500008", "path of pretrained skipthought model checkpoint")
tf.flags.DEFINE_string("book_data_dir", "/media/VSlab3/kuanchen_arxiv/BookCorpus_passage/", "path to directory of book data")
tf.flags.DEFINE_string("book_category", "Adventure", "cateogry of book")
tf.flags.DEFINE_string("style_length_cut_long", 100, "passage which is greather than this length is consider long")
tf.flags.DEFINE_string("style_length_cut_short", 50, "passage which is shorter than this lenght is consider as short ")

config_hardware = tf.ConfigProto()
config_hardware.gpu_options.per_process_gpu_memory_fraction = 0.1
os.environ["CUDA_VISIBLE_DEVICES"]="7"
tf.logging.set_verbosity(tf.logging.INFO)

def load_book_vocab(book_file, num_sample_passage):

    tf.logging.info("Reading original book from %s", book_file)
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

    sentence_long = []
    sentence_short = []
    HACK_CUT = 610 #This is used to remove the skipthought issue reported in the Issue session
    for p in book_sample_passage:
        if len(p) > HACK_CUT:
            continue
        elif len(p.split(" ")) >= FLAGS.style_length_cut_long:
            sentence_long.append(p)
        
        elif len(p.split(" ")) <= FLAGS.style_length_cut_short:
            sentence_short.append(p)

    min_len_long = min([len(e.split(" ")) for e in sentence_long])
    max_len_short = max([len(e.split(" ")) for e in sentence_short])

    assert min_len_long >= FLAGS.style_length_cut_long and max_len_short <= FLAGS.style_length_cut_short

    tf.logging.info("Number of passage long: %s", len(sentence_long))
    tf.logging.info("Number of passage short: %s", len(sentence_short))

    tf.logging.info("loading encoder")
    encoder = encoder_manager.EncoderManager(config=config_hardware)
    encoder.load_model(model_config=stv_config(),
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

    with open("style_bias_bi_skip/bias_" + FLAGS.book_category + "_" + str(FLAGS.style_length_cut_long) + ".pkl", 'w') as handle:
        cPickle.dump(style_bias_long, handle)
    with open("style_bias_bi_skip/bias_" + FLAGS.book_category + "_" + str(FLAGS.style_length_cut_short) + ".pkl", 'w') as handle:
        cPickle.dump(style_bias_short, handle)

if __name__ == "__main__":
    tf.app.run()

