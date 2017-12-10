import tensorflow as tf
def prefetch_input_data(reader,
                        filename,
                        batch_size,
                        values_per_shard,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="fifo_input_queue"):
    
    filename_queue = tf.train.string_input_producer([filename],shuffle=True, capacity=16, name=shard_queue_name)
    
    capacity = values_per_shard + 3 * batch_size
    values_queue = tf.FIFOQueue(capacity=capacity,dtypes=[tf.string],shapes=[[]],name=value_queue_name)
    
    enqueue_ops = []
    
    for _ in range(num_reader_threads):
        
        _, value = reader.read(filename_queue)
        enqueue_ops.append(values_queue.enqueue([value]))
    
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(values_queue, enqueue_ops))
    
    return values_queue

def parse_sequence_example(serialized, code_feature, text_feature):
    context, sequence = tf.parse_single_sequence_example(
      serialized,
      context_features={
          code_feature: tf.FixedLenFeature([2400], dtype=tf.float32)
      },
      sequence_features={
          text_feature: tf.FixedLenSequenceFeature([], dtype=tf.int64),
      })

    text_code = context[code_feature]
    text = sequence[text_feature]
    return text_code, text

def batch_with_dynamic_pad(code_and_text, batch_size, queue_capacity):

    #This style is for multithread extension in the future, we assume single thread here
    enqueue_list = []
    for code, text in code_and_text:
        text_length = tf.shape(text)[0]
        input_length = tf.expand_dims(tf.subtract(text_length, 1), 0)
        
        input_seq = tf.slice(text, [0], input_length)
        target_seq = tf.slice(text, [1], input_length)
        indicator = tf.ones(input_length, dtype=tf.int32)
        enqueue_list.append([code, input_seq, target_seq, indicator])
        #enqueue_list.append([code, input_seq, indicator])
        
    code, input_seqs, target_seqs, mask = tf.train.batch(
        enqueue_list[0],
        batch_size=batch_size,
        capacity=queue_capacity,
        dynamic_pad=True,
        name="batch_and_pad")
#    code, input_seqs, mask = tf.train.batch(
#        enqueue_list[0],
#        batch_size=batch_size,
#        capacity=queue_capacity,
#        dynamic_pad=True,
#        name="batch_and_pad")
    return code, input_seqs, target_seqs, mask
#    return code, input_seqs, mask

"""
def parse_example_batch(serialized, batch_size):

    features = tf.parse_example(
      serialized,
      features={
          "encodings": tf.VarLenFeature(dtype=tf.float32),
          "text": tf.VarLenFeature(dtype=tf.int64),
      })
    
    code = tf.sparse_tensor_to_dense(features["encodings"])
    text = tf.sparse_tensor_to_dense(features["text"])
    mask = tf.sparse_to_dense(features["text"].indices, features["text"].dense_shape, \
                             tf.ones_like(features["text"].values, dtype=tf.int32))
    
    text_len = tf.shape(text)[1]
    input_text = tf.slice(text, [0,0],[batch_size, tf.subtract(text_len,1)])
    target_text = tf.slice(text, [0, 1], [batch_size, tf.subtract(text_len,1)])
    target_mask = tf.slice(mask, [0, 1], [batch_size, tf.subtract(text_len,1)])
    
    return code, input_text, target_text, target_mask
"""
