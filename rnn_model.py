import tensorflow as tf
from data_utils import PrepareClassifyData


class RnnModel(object):
    def __init__(self, conf):
        self._config = conf
        self._init_placeholder()
        self._embedding_layers()
        self._inference()
        self._build_train_op()
        self.sess = tf.Session()
        self.checkpointDir = "model/bi_lstm/"

    def _init_placeholder(self):
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_x")
        self.targets = tf.placeholder(tf.int32, [None], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    def _embedding_layers(self):
        with tf.variable_scope(name_or_scope="embedding_layers"):
            embedding_matrix = tf.get_variable(
                name="embedding_matrix", shape=[self._config.vocab_size, self._config.embedding_size], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
            )
            self.embedded_inputs = tf.nn.embedding_lookup(params=embedding_matrix, ids=self.inputs)

    def _inference(self):
        with tf.variable_scope("bi_lstm"):
            self.sequence_length = tf.reduce_sum(
                tf.cast(tf.not_equal(tf.cast(0, self.inputs.dtype), self.inputs), tf.int32), axis=-1)
            cell_fw = tf.nn.rnn_cell.LSTMCell(self._config.num_hidden)
            cell_bw = tf.nn.rnn_cell.LSTMCell(self._config.num_hidden)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.embedded_inputs, sequence_length=self.sequence_length,
                dtype=tf.float32, time_major=False
            )
            bi_outputs = tf.nn.dropout(tf.concat([output_fw, output_bw], axis=-1), self.keep_prob)
            last_state = bi_outputs[:, -1, :]
        with tf.variable_scope("dense"):
            outputs = tf.layers.dense(last_state, self._config.dense_units, activation=tf.nn.tanh)
        with tf.variable_scope("logits"):
            w = tf.get_variable(name="w", shape=[self._config.dense_units, self._config.num_classes],
                                dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(name="b", shape=[self._config.num_classes], dtype=tf.float32)
            self.logits = tf.matmul(outputs, w, name="logits") + b
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

    def _build_train_op(self):
        with tf.variable_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets)
            self.loss = tf.reduce_mean(losses)
        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(tf.cast(self.predictions, tf.int32), self.targets)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
        self.train_op = tf.train.AdamOptimizer(self._config.learning_rate).minimize(self.loss)

    def _save(self):
        if not tf.gfile.Exists(self.checkpointDir):
            tf.gfile.MakeDirs(self.checkpointDir)
        saver = tf.train.Saver()
        saver.save(sess=self.sess, save_path=self.checkpointDir)

    def _load(self):
        saver = tf.train.Saver()
        saver.restore(sess=self.sess, save_path=tf.train.latest_checkpoint(self.checkpointDir))

    def train(self, flag):
        self.sess.run(tf.global_variables_initializer())
        print("\nbegin train ....\n")
        step = 0
        for i in range(flag.epoch):
            trainset = PrepareClassifyData(flag, "train", False)
            for input_x, input_y in trainset:
                print(input_x.shape)
                step += (i+1) * len(input_y)
                _, loss, acc = self.sess.run(
                    fetches=[self.train_op, self.loss, self.accuracy],
                    feed_dict={self.inputs: input_x, self.targets: input_y, self.keep_prob: 0.5})
                print("<Train>\t Epoch: [%d] Iter: [%d] Step: [%d] Loss: [%.3F]\t Acc: [%.3f]" %
                      (i+1, int(step/flag.batch_size), step, loss, acc))
            self._save()

    def test(self, flag):
        print("\nbegin test ....\n")
        _iter = 0
        testset = PrepareClassifyData(flag, "test", False)
        for input_x, input_y in testset:
            _iter += 1
            acc, loss = self.sess.run(
                fetches=[self.accuracy, self.loss],
                feed_dict={self.inputs: input_x, self.targets: input_y, self.keep_prob: 1.})
            print("<Test>\t Iter: [%d] Loss: [%.3F]\t Acc: [%.3f]" %
                  (_iter, loss, acc))
