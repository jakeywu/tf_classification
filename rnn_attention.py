import tensorflow as tf


class BaseModel(object):
    def __init__(self):
        self.sess = tf.Session()
        self.checkpointDir = "model/checkpoints/"

    def _save(self):
        saver = tf.train.Saver()
        saver.save(sess=self.sess, save_path="{}-classification".format(self.checkpointDir))

    def _load(self):
        saver = tf.train.Saver()
        saver.restore(sess=self.sess, save_path=tf.train.latest_checkpoint(self.checkpointDir))


class RnnAttention(BaseModel):
    def __init__(self, conf):
        super(RnnAttention, self).__init__()
        self.num_classes = conf.num_classes
        self.vocab_size = conf.vocab_size
        self.embedding_size = conf.embedding_size
        self.num_hidden = conf.num_hidden
        self.word_attention_size = conf.attention_size
        self.sentence_attention_size = conf.sentence_attention_size

    def _placeholder_layers(self):
        # [BatchSize * Sentence * Word]
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name="inputs")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None], name="targets")
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name="keep_prob")

        # The true sequence_length of word about every sentence.  [BatchSize * Sentence]
        self.word_length = tf.reduce_sum(
            tf.cast(tf.not_equal(tf.cast(0, self.inputs.dtype), self.inputs), tf.int32), axis=-1
        )
        # The true sequence_length of sentence about every article. [BatchSize]
        self.sentence_length = tf.reduce_sum(
            tf.cast(tf.not_equal(tf.cast(0, self.inputs.dtype), self.word_length), tf.int32), axis=-1
        )

    def _embedding_layers(self):
        with tf.variable_scope(name_or_scope="embedding_layers"):
            embedding_matrix = tf.get_variable(
                name="embedding_matrix", shape=[self.vocab_size, self.embedding_size], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
            )
            self.embedded_inputs = tf.nn.embedding_lookup(params=embedding_matrix, ids=self.inputs)
            # [B * S * W * D]
            self.origin_shape = tf.shape(self.embedded_inputs)

    def __word_encoder_layers(self):
        with tf.variable_scope(name_or_scope="word_encoder_layers"):
            cell_fw = tf.nn.rnn_cell.GRUCell(num_units=self.num_hidden)
            cell_bw = tf.nn.rnn_cell.GRUCell(num_units=self.num_hidden)
            word_level_inputs = tf.reshape(self.embedded_inputs, [
                self.origin_shape[0] * self.origin_shape[1], self.origin_shape[2], self.embedding_size])

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=word_level_inputs, sequence_length=self.word_length,
                dtype=tf.float32, time_major=False
            )
            self.word_encoder_output = tf.nn.dropout(x=tf.concat([output_fw, output_bw], axis=2), keep_prob=self.keep_prob)

    def _word_attention_layers(self):
        with tf.variable_scope("word_attention_layers"):
            attention_context_vector = tf.get_variable(
                name="attention_context_vector", shape=[self.word_attention_size],
                initializer=tf.constant_initializer(0.)
            )
            # [BatchSize * Sentence, Word, WordAttentionSize]
            input_projection = tf.layers.dense(inputs=self.word_encoder_output, units=self.word_attention_size, activation=tf.tanh)
            vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keep_dims=True)

            alphas = tf.nn.softmax(vector_attn, dim=1)
            weighted_projection = tf.multiply(input_projection, alphas)
            self.word_attention_output = tf.reduce_sum(weighted_projection, axis=1)

    def train(self):
        pass

    def test(self):
        pass

    def validate(self):
        pass
