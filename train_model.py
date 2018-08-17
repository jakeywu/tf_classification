import tensorflow as tf

tf.flags.DEFINE_string(name="model", default="cnn", help="selected in [rnn, cnn, rnn_attention]")

FLAG = tf.flags.FLAGS


class BaseInitHyperParams(object):
    epoch = 2  # 轮次
    batch_size = 64  # 批次
    vocab_size = 5000  # 字数
    embedding_size = 128  # 字向量维度
    learning_rate = 1e-3  # 学习速率
    num_classes = 10  # 类别树目
    classify_names = "体育,财经,房产,家居,教育,科技,时尚,时政,游戏,娱乐"  # 类别名称/具体根据语料设定
    max_document_length = 600  # 文章首位字符数量, 考虑新闻文章首位重要性, 选择 2 * max_document_length数量


class RnnAttentionParams(BaseInitHyperParams):
    """
    Bi-GRU + Attention
    """
    word_num_hidden = 64  # 字神经元个数
    sentence_num_hidden = 64  # 句子神经元个数
    word_attention_size = 64  # 字注意力
    sentence_attention_size = 32  # 句子注意力


class RnnParams(BaseInitHyperParams):
    num_hidden = 128
    dense_units = 128


class CnnParams(BaseInitHyperParams):
    num_filters = 64
    sequence_length = 1200
    filter_sizes = [3, 6, 9]


def main(_):
    if FLAG.model == "cnn":
        params = CnnParams()
        from cnn_model import CnnModel
        model = CnnModel(params)
    elif FLAG.model == "rnn":
        params = RnnParams()
        from rnn_model import RnnModel
        model = RnnModel(params)
    elif FLAG.model == "rnn_attention":
        from rnn_attention_model import RnnAttentionModel
        params = RnnAttentionParams()
        model = RnnAttentionModel(params)
    else:
        raise Exception("model only can be in [cnn, rnn, rnn_attention]")
    model.train(params)
    model.test(params)


if __name__ == "__main__":
    tf.app.run(main)
