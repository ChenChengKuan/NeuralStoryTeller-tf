import tensorflow as tf
import op.input_teller as input_ops

class StoryDecoder():
    
    def __init__(self, config, mode):
        
        assert mode in ["train", "inference"]
        self.config = config
        self.mode = mode
        
        #TF record reader
        self.reader = tf.TFRecordReader()
        
        #TODO: Should be refactor to combined embedding extraction later
        
        # A float32 Tensor with shape [batch_size, padded_length, embedding_size]. 
        #This is the original embedding
        self.st_embed_orig = None
        
        #This is the finalized (reduced:2400->800) version of embedding
        self.skip_thought_embeddings = None


        self.initializer = tf.random_uniform_initializer(minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)
        
        self.seq_embeddings = None
    
        self.embedding_map = tf.get_variable(name="embed_map",shape=[self.config.vocab_size, self.config.w2v_embedding_size],
          initializer=self.initializer)
            
        self.pre_embed = None
        
        self.input_seqs = None
        
        self.input_mask = None
        
        self.target_seqs = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_losses = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_loss_weights = None

        # Global step Tensor.
        self.global_step = None
    
    def build_inputs(self):

        if self.mode == "inference": 
            st_embed_orig = tf.placeholder(dtype=tf.float32, shape=[None,self.config.sk_embedding_size], name="st_feed")
            input_feed = tf.placeholder(dtype=tf.int64, shape=[None], name="input_feed")
            input_seq = tf.expand_dims(input_feed, 1)
            target_seq = None
            mask = None

        else:
            input_queue = input_ops.prefetch_input_data(self.reader, 
                                                       self.config.filename,
                                                        self.config.batch_size,
                                                       values_per_shard=self.config.values_per_input_shard)
            code_and_text = []
            serialized_sequence_example = input_queue.dequeue()
            text_code, text = input_ops.parse_sequence_example(serialized_sequence_example,
                                             code_feature=self.config.code_name,
                                             text_feature=self.config.text_name)

            code_and_text.append([text_code, text])
            queue_capacity = (2 * self.config.num_preprocess_threads * self.config.batch_size)
            
            st_embed_orig,input_seq, target_seq, mask = input_ops.batch_with_dynamic_pad(code_and_text=code_and_text,
                                                                                         batch_size=self.config.batch_size,
                                                                                        queue_capacity=queue_capacity)
            #st_embed_orig, input_seq, mask = input_ops.batch_with_dynamic_pad(code_and_text=code_and_text,
            #                                                        batch_size=32,
            #                                                        queue_capacity=queue_capacity)
            
        self.st_embed_orig = st_embed_orig
        self.input_seqs = input_seq
        self.target_seqs = target_seq
        self.input_mask = mask
        
    def build_seq_embeddings(self):
        
        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            seq_embeddings = tf.nn.embedding_lookup(self.embedding_map, self.input_seqs)

        self.seq_embeddings = seq_embeddings
        
    def build_skipthought_embedding(self):
        
        with tf.variable_scope("skipthought_embedding") as scope:
            
            skip_thought_embeddings = tf.contrib.layers.fully_connected(
                inputs=self.st_embed_orig,
                num_outputs=self.config.lstm_embedding_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None,
                scope=scope)
        
        self.skip_thought_embeddings = skip_thought_embeddings
    
    def build_model(self):
        
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.config.num_lstm_units, state_is_tuple=True)
        
        if self.mode == "train":
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell,
                input_keep_prob=self.config.lstm_dropout_keep_prob,
                output_keep_prob=self.config.lstm_dropout_keep_prob)
        
            
        with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
            #Feed the sequence embeddings to set the initial LSTM state.
            if self.mode == "train":
                zero_state = lstm_cell.zero_state(batch_size=self.skip_thought_embeddings.get_shape()[0],dtype=tf.float32)

            elif self.mode == "inference":
                zero_state = lstm_cell.zero_state(batch_size=tf.placeholder(tf.int32, [], name="test_batch_size"), dtype=tf.float32)
            _, initial_state = lstm_cell(self.skip_thought_embeddings, zero_state)

            # Allow the LSTM variables to be reused.
            lstm_scope.reuse_variables()
            if self.mode == 'inference':
                # In inference mode, use concatenated states for convenient feeding and
                # fetching.
                tf.concat(axis=1, values=initial_state, name="initial_state")

                # Placeholder for feeding a batch of concatenated states.
                state_feed = tf.placeholder(dtype=tf.float32,
                                            shape=[None, sum(lstm_cell.state_size)],
                                            name="state_feed")
                state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

                # Run a single LSTM step.
                lstm_outputs, state_tuple = lstm_cell(
                    inputs=tf.squeeze(self.seq_embeddings, axis=[1]),
                    state=state_tuple)

                # Concatentate the resulting state.
                tf.concat(axis=1, values=state_tuple, name="state")
            else:
                # Run the batch of sequence embeddings through the LSTM.
                sequence_length = tf.reduce_sum(self.input_mask, 1)
                lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                    inputs=self.seq_embeddings,
                                                    sequence_length=sequence_length,
                                                    initial_state=initial_state,
                                                    dtype=tf.float32)

            lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

        with tf.variable_scope("logits") as logits_scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=lstm_outputs,
                num_outputs=self.config.vocab_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                scope=logits_scope)

        if self.mode == "inference":
            tf.nn.softmax(logits, name="softmax")
        else:
            targets = tf.reshape(self.target_seqs, [-1])
            weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

            # Compute losses.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                                  logits=logits)
            batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                              tf.reduce_sum(weights),
                              name="batch_loss")
            tf.losses.add_loss(batch_loss)
            total_loss = tf.losses.get_total_loss()

            # Add summaries.
            tf.summary.scalar("losses/batch_loss", batch_loss)
            tf.summary.scalar("losses/total_loss", total_loss)
            for var in tf.trainable_variables():
                tf.summary.histogram("parameters/" + var.op.name, var)

            self.total_loss = total_loss
            self.target_cross_entropy_losses = losses  # Used in evaluation.
            self.target_cross_entropy_loss_weights = weights  # Used in evaluation.
            
    def setup_global_step(self):
         """Sets up the global step Tensor."""
         global_step = tf.Variable(
             initial_value=0,
             name="global_step",
             trainable=False,
             collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

         self.global_step = global_step

    def build(self):
         """Creates all ops for training and evaluation."""
         self.build_inputs()
         self.build_skipthought_embedding()
         self.build_seq_embeddings()
         self.build_model()
         self.setup_global_step()
