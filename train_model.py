import tensorflow as tf
from rnn_attention import RnnAttention

tf.flags.DEFINE_integer(name="epoch", default=5, help="maximum epochs")
tf.flags.DEFINE_integer(name="batch_size", default=64, help="batch size")
tf.flags.DEFINE_integer(name="vocab_size", default=5000, help="vocab num of chinese")
tf.flags.DEFINE_integer(name="embedding_size", default=128, help="embedding size")
tf.flags.DEFINE_integer(name="num_hidden", default=128, help="lstm num hidden")

FLAG = tf.flags.FLAGS


def main(_):
    model = RnnAttention(FLAG)
    model.train()
    model.test()


if __name__ == "__main__":
    tf.app.run(main)
