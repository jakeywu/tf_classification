import tensorflow as tf
from rnn_attention import RnnAttention

tf.flags.DEFINE_integer(name="epoch", default=5, help="maximum epochs")
tf.flags.DEFINE_integer(name="batch_size", default=64, help="batch size")
tf.flags.DEFINE_integer(name="vocab_size", default=5000, help="vocab num of chinese")
tf.flags.DEFINE_integer(name="embedding_size", default=128, help="embedding size")
tf.flags.DEFINE_integer(name="word_num_hidden", default=128, help="lstm num hidden")
tf.flags.DEFINE_integer(name="sentence_num_hidden", default=64, help="lstm num hidden")
tf.flags.DEFINE_integer(name="num_classes", default=10, help="num of categories")
tf.flags.DEFINE_float(name="learning_rate", default=1e-3, help="learning rate ")

tf.flags.DEFINE_integer(name="word_attention_size", default=30, help="word attention size")
tf.flags.DEFINE_integer(name="sentence_attention_size", default=10, help="sentence attention size")

tf.flags.DEFINE_string(name="classify_names", default="体育,财经,房产,家居,教育,科技,时尚,时政,游戏,娱乐", help="category tags")

FLAG = tf.flags.FLAGS


def main(_):
    model = RnnAttention(FLAG)
    model.train(FLAG)
    model.test(FLAG)


if __name__ == "__main__":
    tf.app.run(main)
