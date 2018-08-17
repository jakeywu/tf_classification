from data_utils import PrepareClassifyData
import tensorflow as tf


class BaseModel(object):
    def __init__(self):
        self.sess = tf.Session()
        self.checkpointDir = "model/"

    def _save(self):
        saver = tf.train.Saver()
        saver.save(sess=self.sess, save_path="{}-rnn-attention".format(self.checkpointDir))

    def _load(self):
        saver = tf.train.Saver()
        saver.restore(sess=self.sess, save_path=tf.train.latest_checkpoint(self.checkpointDir))


class RnnAttentionModel(BaseModel):
    def __init__(self, conf):
        super(RnnAttentionModel, self).__init__()
        self.epoch = conf.epoch
        self.num_classes = conf.num_classes
        self.vocab_size = conf.vocab_size
        self.learning_rate = conf.learning_rate
        self.embedding_size = conf.embedding_size
        self.word_num_hidden = conf.word_num_hidden
        self.word_attention_size = conf.word_attention_size
        self.sentence_num_hidden = conf.sentence_num_hidden
        self.sentence_attention_size = conf.sentence_attention_size
        self._placeholder_layers()
        self._embedding_layers()
        self._word_encoder_layers()
        self._word_attention_layers()
        self._sentence_encoder_layers()
        self._sentence_attention_layers()
        self._inference()
        self._build_train_op()

    def _placeholder_layers(self):
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name="inputs")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None], name="targets")
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name="keep_prob")

        self.word_length = tf.reduce_sum(
            tf.cast(tf.not_equal(tf.cast(0, self.inputs.dtype), self.inputs), tf.int32), axis=-1
        )
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

    def _word_encoder_layers(self):
        with tf.variable_scope(name_or_scope="word_encoder_layers"):
            cell_fw = tf.nn.rnn_cell.GRUCell(num_units=self.word_num_hidden)
            cell_bw = tf.nn.rnn_cell.GRUCell(num_units=self.word_num_hidden)
            word_inputs = tf.reshape(
                self.embedded_inputs, [self.origin_shape[0] * self.origin_shape[1], self.origin_shape[2], self.embedding_size])
            word_length = tf.reshape(self.word_length, [self.origin_shape[0] * self.origin_shape[1]])
            (output_fw, output_bw), (a, b) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=word_inputs, sequence_length=word_length,
                dtype=tf.float32, time_major=False
            )
            import pdb
            pdb.set_trace()
            self.word_encoder_output = tf.nn.dropout(x=tf.concat([output_fw, output_bw], axis=2), keep_prob=self.keep_prob)

    def _word_attention_layers(self):
        with tf.variable_scope("word_attention_layers"):
            w_1 = tf.get_variable(
                name="w_1", shape=[2 * self.word_num_hidden, self.word_attention_size],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            b_1 = tf.get_variable(name="b_1", shape=[self.word_attention_size], initializer=tf.constant_initializer(0.))
            u = tf.get_variable(
                name="w_2", shape=[self.word_attention_size, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
            v = tf.nn.xw_plus_b(tf.reshape(self.word_encoder_output, [-1, 2 * self.word_num_hidden]), w_1, b_1)  # B*T*A
            s = tf.matmul(tf.nn.tanh(v), u)
            alphas = tf.nn.softmax(tf.reshape(s, [self.origin_shape[0] * self.origin_shape[1], 1, self.origin_shape[2]]))
            self.word_attention_output = tf.reduce_sum(tf.matmul(alphas, self.word_encoder_output), axis=1)

    def _sentence_encoder_layers(self):
        with tf.variable_scope(name_or_scope="sentence_encoder_layers"):
            cell_fw = tf.nn.rnn_cell.GRUCell(num_units=self.sentence_num_hidden)
            cell_bw = tf.nn.rnn_cell.GRUCell(num_units=self.sentence_num_hidden)
            sentence_level_inputs = tf.reshape(self.word_attention_output, [
                self.origin_shape[0], self.origin_shape[1], 2 * self.word_num_hidden])

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=sentence_level_inputs,
                sequence_length=self.sentence_length,
                dtype=tf.float32, time_major=False
            )
            self.sentence_encoder_output = tf.nn.dropout(x=tf.concat([output_fw, output_bw], axis=2),
                                                         keep_prob=self.keep_prob)

    def _sentence_attention_layers(self):
        with tf.variable_scope("sentence_attention_layers"):
            w_1 = tf.get_variable(
                name="w_1", shape=[2 * self.sentence_num_hidden, self.sentence_attention_size],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            b_1 = tf.get_variable(name="b_1", shape=[self.sentence_attention_size], initializer=tf.constant_initializer(0.))
            u = tf.get_variable(
                name="w_2", shape=[self.sentence_attention_size, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
            v = tf.nn.xw_plus_b(tf.reshape(self.sentence_encoder_output, [-1, 2 * self.sentence_num_hidden]), w_1, b_1)  # B*T*A
            s = tf.matmul(v, u)
            alphas = tf.nn.softmax(tf.reshape(s, [self.origin_shape[0], 1, self.origin_shape[1]]))
            self.sentence_attention_output = tf.reduce_sum(tf.matmul(alphas, self.sentence_encoder_output), axis=1)

    def _inference(self):
        with tf.variable_scope("train_op"):
            w = tf.get_variable(
                name="w", shape=[2 * self.sentence_num_hidden, self.num_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(
                name="b", shape=[self.num_classes], initializer=tf.constant_initializer(0.)
            )
            self.logits = tf.matmul(self.sentence_attention_output, w) + b
            self.predictions = tf.argmax(self.logits, axis=1)
            correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.targets)
            self.accuracy_val = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")

    def _build_train_op(self):
        self.total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=self.logits)
        self.loss = tf.reduce_mean(self.total_loss)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    def train(self, flag):
        self.sess.run(tf.global_variables_initializer())
        print("\nbegin train ....\n")
        step = 0
        for i in range(self.epoch):
            trainset = PrepareClassifyData(flag, "train", True)
            for input_x, input_y in trainset:
                step += (i+1) * len(input_y)
                _, loss, acc = self.sess.run(
                    fetches=[self.train_op, self.loss, self.accuracy_val],
                    feed_dict={self.inputs: input_x, self.targets: input_y, self.keep_prob: 0.5})
                print("<Train>\t Epoch: [%d] Iter: [%d] Step: [%d] Loss: [%.3F]\t Acc: [%.3f]" %
                      (i+1, int(step/flag.batch_size), step, loss, acc))
            self._save()

    def test(self, flag):
        print("\nbegin test ....\n")
        _iter = 0
        testset = PrepareClassifyData(flag, "test")
        for input_x, input_y in testset:
            _iter += 1
            acc, loss = self.sess.run(
                fetches=[self.accuracy_val, self.loss],
                feed_dict={self.inputs: input_x, self.targets: input_y, self.keep_prob: 1.})
            print("<Test>\t Iter: [%d] Loss: [%.3F]\t Acc: [%.3f]" %
                  (_iter, loss, acc))
