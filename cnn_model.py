import tensorflow as tf
from data_utils import PrepareClassifyData


class CnnModel(object):
    def __init__(self, conf):
        self._config = conf
        self._init_placeholder()
        self._embedding_layers()
        self._inference()
        self._build_train_op()
        self.checkpointDir = "model/cnn/"
        self.sess = tf.Session()

    def _init_placeholder(self):
        self.inputs = tf.placeholder(tf.int32, [None, self._config.sequence_length], name="input_x")
        self.targets = tf.placeholder(tf.int32, [None], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    def _embedding_layers(self):
        with tf.variable_scope(name_or_scope="embedding_layers"):
            embedding_matrix = tf.get_variable(
                name="embedding_matrix", shape=[self._config.vocab_size, self._config.embedding_size], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
            )
            self.embedded_inputs = tf.nn.embedding_lookup(params=embedding_matrix, ids=self.inputs)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_inputs, 1)

    def _inference(self):
        self.total_features = len(self._config.filter_sizes) * self._config.num_filters
        self._cnn_layers()
        with tf.variable_scope("score"):
            w = tf.get_variable(name="w", shape=[self.total_features, self._config.num_classes],
                                dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(name="b", shape=[self._config.num_classes], dtype=tf.float32)
            self.logits = tf.matmul(self._dropout_pool_flat, w, name="logits") + b
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

    def _build_train_op(self):
        with tf.variable_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets)
            self.loss = tf.reduce_mean(losses)
        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(tf.cast(self.predictions, tf.int32), self.targets)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self._config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    def _cnn_layers(self):
        pooled_outputs = []
        for filter_size in self._config.filter_sizes:
            with tf.variable_scope("cnn-%s" % str(filter_size)):
                conv1 = self._cnn_2d(
                    self.embedded_chars_expanded, "1", self._config.embedding_size, self._config.num_filters, 1,
                    filter_size)
                relu1 = tf.nn.relu(conv1)
                pool1 = self._max_pool(relu1, self._config.sequence_length - filter_size + 1)
                pooled_outputs.append(pool1)

        pool_flat = tf.reshape(tf.concat(pooled_outputs, 3), [-1, self.total_features])
        with tf.variable_scope("dropout"):
            self._dropout_pool_flat = tf.nn.dropout(pool_flat, self.keep_prob)

    @staticmethod
    def _cnn_2d(neural, scope_name, in_channels, out_channels, filter_height, filter_width):
        """二维图像卷积"""
        with tf.variable_scope(name_or_scope=scope_name):
            kernel = tf.get_variable(
                name="W", shape=[filter_height, filter_width, in_channels, out_channels], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.01)
            )
            bias = tf.get_variable(name="a", shape=[out_channels], dtype=tf.float32,
                                   initializer=tf.constant_initializer())
            con2d_op = tf.nn.conv2d(input=neural, filter=kernel, strides=[1, 1, 1, 1], padding="VALID")
        return tf.nn.bias_add(con2d_op, bias=bias)

    @staticmethod
    def _max_pool(neural, width_ksize):
        return tf.nn.max_pool(
            neural, ksize=(1, 1, width_ksize, 1), strides=[1, 1, 1, 1], padding="VALID", name="max_pool"
        )

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
