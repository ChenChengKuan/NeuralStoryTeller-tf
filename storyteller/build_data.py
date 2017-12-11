from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('..')
import tensorflow as tf
import collections
import numpy as np
import configuration
import os
import nltk
from skipthought import encoder_manager
np.random.seed(99)

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("stv_vocab", "/media/VSlab3/kuanchen_arxiv/skip_thoughts_uni_2017_02_02/vocab.txt", "vocab used by stv (expanded)")
tf.flags.DEFINE_string("stv_embedding", "/media/VSlab3/kuanchen_arxiv/skip_thoughts_uni_2017_02_02/embeddings.npy", "stv embeddings of vocabulary")
tf.flags.DEFINE_string("stv_model", "/media/VSlab3/kuanchen_arxiv/skip_thoughts_uni_2017_02_02/model.ckpt-501424", "checkpoint of pretrained skipthought model")
tf.flags.DEFINE_string("book_data_dir", "/media/VSlab3/kuanchen_arxiv/BookCorpus_passage/", "directory of book data")
tf.flags.DEFINE_string("book_category", "Adventure", "cateogry of book")
tf.flags.DEFINE_string("output_dir", "/media/VSlab3/kuanchen_arxiv/vocab_60000_advent_allpad", "output directoy")

config_hardware = tf.ConfigProto()
config_hardware.gpu_options.per_process_gpu_memory_fraction = 0.2
os.environ["CUDA_VISIBLE_DEVICES"]="2"
tf.logging.set_verbosity(tf.logging.INFO)

def load_expanded_vocab(vocab_file):
    """load the expanded version of vocabulary file
    
    Args:
        vocab_file: vocab_ext.txt file which store the vocabulary line by line

    Returns:
        vocab: An ordered dictionary of word to id
    """ 
    tf.logging.info("Reading expanded vocabulary from %s", vocab_file)
    vocab = collections.OrderedDict()
    with tf.gfile.GFile(vocab_file, mode="r") as f:
        for i,line in enumerate(f):
            word = line.decode("utf-8").strip()
            if word in vocab:
                continue
            else:
                vocab[word] = i
    tf.logging.info("Read vocab of size %d from %s",len(vocab), vocab_file)
    return vocab

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
    
    vocab = collections.Counter()
    
    for p in book_sample_passage:
        vocab.update(p.split(" "))
    
    vocab_mc = vocab.most_common(most_common)
    
    return [w[0] for w in vocab_mc], book_sample_passage      

def _int64_feature(value):
  """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def _sentence_to_ids(sentence, vocab, unk_id, sos_id, eos_id):
    """Helper function to convert words in sentence to ids, with insertion of special <sos> and <eos> id"""
    add_eos = True #A parameter feeded from outside, should be refactor later
    add_sos = True
    assert add_sos == True and add_eos == True
    ids = [vocab.get(w, unk_id) for w in sentence]
    if add_eos:
        ids.append(eos_id)
    if add_sos:
        ids.insert(0,sos_id)
    return ids


def main(unused_arg):
    vocab_book, book_sample_passage = load_book_vocab(FLAGS.book_data_dir + FLAGS.book_category + "/*",\
                                                      most_common=60000, \
                                                      num_sample_passage=1000)
    vocab_exp = load_expanded_vocab(FLAGS.stv_vocab)
    with open(FLAGS.stv_embedding, 'r') as f:
        embedding_matrix = np.load(f)
    tf.logging.info("Loaded embedding matrix with shape %s",embedding_matrix.shape)

    ####
    #This section left blan for source corpus expansion (default coco caption)
    ####
    vocab_book_augment = ['<eos>','<unk>']
    vocab_book_augment.extend(vocab_book)
    vocab_embd_idx_book = []
    for v in vocab_book_augment:
        if v in vocab_exp:
            vocab_embd_idx_book.append(vocab_exp[v])
        else:
            vocab_embd_idx_book.append(vocab_exp['<unk>'])
    assert len(vocab_book_augment) == len(vocab_book) + 2
    embedding_matrix_reduced = embedding_matrix[list(vocab_embd_idx_book)]
    
    #Add a randomly initialized embedding array for start and end token
    start_token_embed = np.random.randn(1,embedding_matrix.shape[1])[0]
    embedding_matrix_reduced = np.vstack([embedding_matrix_reduced, start_token_embed/np.linalg.norm(start_token_embed)])
    
    vocab_book_augment.append('<sos>')
    assert len(vocab_book_augment) == len(vocab_book) + 3
    tf.logging.info("Create a reduced embedding matrix with shape %s", embedding_matrix_reduced.shape)
    embeddings_reduced_file = os.path.join(FLAGS.output_dir, "embeddings_r.npy")
    np.save(embeddings_reduced_file, embedding_matrix_reduced)
    tf.logging.info("Wrote reduced embeddings file to %s", embeddings_reduced_file)

    vocab_file_path = os.path.join(FLAGS.output_dir, "vocab_shared.txt")
    with tf.gfile.FastGFile(vocab_file_path, "w") as f:
        f.write("\n".join(vocab_book_augment))
    tf.logging.info("Wrote vocab plus start to %s", vocab_file_path)

    vocab_book_augment_dict = collections.OrderedDict()
    for i,v in enumerate(vocab_book_augment):
        vocab_book_augment_dict[v] = i
    assert np.array_equal(embedding_matrix[vocab_exp['the']], embedding_matrix_reduced[vocab_book_augment_dict['the']])
    assert np.array_equal(start_token_embed/np.linalg.norm(start_token_embed), embedding_matrix_reduced[vocab_book_augment_dict['<sos>']])

    max_word_len = 100
    passage_overlength = 0
    book_sample_passage_cut = []

    for p in book_sample_passage:
        if len(p.split(" ")) > max_word_len:
            passage_overlength += 1
            continue
        else:
            book_sample_passage_cut.append(p)

    tf.logging.info("Number of passage over length: %s", passage_overlength)
    tf.logging.info("Number of passage : %s", len(book_sample_passage_cut))

    tf.logging.info("loading encoder")
    encoder = encoder_manager.EncoderManager(config=config_hardware)
    encoder.load_model(configuration.stv_config(),
                       vocabulary_file=FLAGS.stv_vocab,
                       embedding_matrix_file=FLAGS.stv_embedding,
                       checkpoint_path=FLAGS.stv_model)
    tf.logging.info("encoding data")
    encodings = encoder.encode(book_sample_passage_cut)
    tf.logging.info("Writing data")

    book_sample_passage_valid = []
    encodings_valid = []
    num_invalid_encoding = 0
    for i,e in enumerate(encodings):
        
        if np.isnan(e).any():
            num_invalid_encoding+=1
            continue
        else:
            encodings_valid.append(e)
            book_sample_passage_valid.append(book_sample_passage_cut[i])

    tf.logging.info("Number of invalid encoding : %s", num_invalid_encoding)
    tf.logging.info("Numeber of valid sample: %s", len(book_sample_passage_valid))
    tf.logging.info("Writing data")
    tf_record_file = os.path.join(FLAGS.output_dir, "stv_text_code.tfrecord")
    writer = tf.python_io.TFRecordWriter(tf_record_file)
    SOS_ID = vocab_book_augment_dict['<sos>']
    EOS_ID = vocab_book_augment_dict['<eos>']
    UNK_ID = vocab_book_augment_dict['<unk>']
    for idx, text in enumerate(book_sample_passage_valid):

        context = tf.train.Features(feature={
          "text_stv": _bytes_feature(encodings[idx])
        })

        feature_lists = tf.train.FeatureLists(feature_list={
          "text": _int64_feature_list(_sentence_to_ids(text.split(" "), vocab_book_augment_dict, unk_id=UNK_ID, sos_id=SOS_ID, eos_id=EOS_ID))
        })
        sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
        writer.write(sequence_example.SerializeToString())
    writer.close()

if __name__ == "__main__":
    tf.app.run()

