import os
import numpy as np
import codecs
import tensorflow as tf


class PrepareData(object):
    def __init__(self, model):
        """
        :param model: cnn, rnn, rnn_attention
        """
        context1 = """
        马晓旭意外受伤让国奥警惕 无奈大雨格外青睐殷家军记者傅亚雨沈阳报道 来到沈阳，国奥队依然没有摆脱雨水的困扰。7月31日下午6点，国奥队的日
        常训练再度受到大雨的干扰，无奈之下队员们只慢跑了25分钟就草草收场。31日上午10点，国奥队在奥体中心外场训练的时候，天就是阴沉沉的，气象
        预报显示当天下午沈阳就有大雨，但幸好队伍上午的训练并没有受到任何干扰。下午6点，当球队抵达训练场时，大雨已经下了几个小时，而且丝毫没有停
        下来的意思。抱着试一试的态度，球队开始了当天下午的例行训练，25分钟过去了，天气没有任何转好的迹象，为了保护球员们，国奥队决定中止当天的
        训练，全队立即返回酒店。在雨中训练对足球队来说并不是什么稀罕事，但在奥运会即将开始之前，全队变得“娇贵”了。在沈阳最后一周的训练，国奥队首
        先要保证现有的球员不再出现意外的伤病情况以免影响正式比赛，因此这一阶段控制训练受伤、控制感冒等疾病的出现被队伍放在了相当重要的位置。而
        抵达沈阳之后，中后卫冯萧霆就一直没有训练，冯萧霆是7月27日在长春患上了感冒，因此也没有参加29日跟塞尔维亚的热身赛。队伍介绍说，冯萧霆并
        没有出现发烧症状，但为了安全起见，这两天还是让他静养休息，等感冒彻底好了之后再恢复训练。由于有了冯萧霆这个例子，因此国奥队对雨中训练就
        显得特别谨慎，主要是担心球员们受凉而引发感冒，造成非战斗减员。而女足队员马晓旭在热身赛中受伤导致无缘奥运的前科，也让在沈阳的国奥队现在
        格外警惕，“训练中不断嘱咐队员们要注意动作，我们可不能再出这样的事情了。”一位工作人员表示。从长春到沈阳，雨水一路伴随着国奥队，“也邪了，
        我们走到哪儿雨就下到哪儿，在长春几次训练都被大雨给搅和了，没想到来沈阳又碰到这种事情。”一位国奥球员也对雨水的“青睐”有些不解。
        """

        context2 = """
        牛年第一月 开基抬头券商集合理财掉队每经记者 于春敏在金融危机的淫威之下，2008年，全球资本市场均经历了一番血雨腥风的洗礼，进入2009年，对大多
        数国家的股市而言，仍然是一片愁云惨淡，看不到放晴迹象，然而，对于A股市场而言，却大有新年新气象、风景这边独好之势。数据显示，在刚刚过去的1月
        份，A股市场一枝独秀，上证指数累计上涨了9.3%，位居全球十大股市之首。伴随着大盘向上发力，开放式基金、券商集合理财产品的净值均出现了久违的上涨。
        整体上看，偏股型券商集合理财产品虽然小有斩获，但是要远远落后于风险相对较高的偏股型基金的阶段表现。开放式基金1月飙涨跌跌不休的A股市场，
        让2008年的开放式基金经历了从云端坠入谷底的惊天大逆转最终以净值缩水1.34万亿元谢幕，让人唏嘘不已。曾被寄予厚望的开放式基金尤其是股票型基金
        何时能触底反弹、东山再起？成了不少投资人心中挥之不去的期待。伴随着新年以来大盘的上涨，人们终于看到了希望。银河证券基金研究中心的统计数据显示，
        1月份，开放式基金迎来了新年的开门红。1月份，股票型、指数型、偏股型和平衡型基金平均上涨6.44%、10.46%、6.29%和4.90%。沪深指数大涨，
        最为受益的莫过于指数型基金。此外，一些投研实力较强的股票型基金也表现突出，其中股票型基金排名前十位的收益率均不小于12%。农历新年后的
        第一个交易日，上证指数更是坚强地站在了2000点之上，各偏股型基金净值快速回升。数据显示，截至2月3日，偏股型基金今年来全部取得正收益，涨幅在
        15%左右的基金随处可见，涨幅超过10%的基金更是比比皆是，连混合偏债型基金都有着3.61%的平均涨幅。根据银河证券的统计，短短一个多月的时间，
        155只股票型基金(剔除新基金)中，前三甲的涨幅均超过了20%，涨幅在16%~19%的基金有16只，28只基金的涨幅在13%~16%之间，而涨幅在10%~13%之间的
        基金更是达到了48只，另有60只基金的涨幅在10%以下。券商集合理财产品暂时落后然而，相比于开放式基金尤其是其中的偏股型基金的凌厉涨势，券商集
        合理财产品则稍微有些“后知后觉”，虽然也出现了不同程度的上涨，但整体而言，显得波澜不惊，明显落后于偏股型基金，且落后于同期大盘。以节前一周的
        表现为例，上证指数当周涨幅1.85%，深证成指上涨1.41%，股票型开放式基金一周净值收益率平均涨幅1.33%、混合型开放式基金净值平均上涨1.3%，封闭
        式基金市价平均下跌0.03%；但当周股票型券商集合理财产品平均收益率仅为0.49%，混合型券商集合理财产品平均净值增长率为1.01%，FOF平均净值增长率
        为0.78%。统计数据显示，截至1月23日，收益最高的偏股型券商集合理财产品当属东方证券旗下的东方红3号，今年来收益为9.83%，国信金理财价值增长和
        广发理财3号分别以5.96%和4.86%的收益率位列二、三位。1月收益率排名靠前的偏股型券商集合理财产品还有中金公司旗下的中金股票精选、中金股票策略、
        中金股票策略二号，累计净值分别为1.8782元、1.0307元和0.763元，成立以来累计净值增长率分别为87.82%、3.07%和-23.7%，今年以来的收益分别为
        2.14%、1.68%和4.42%；另外，上海证券旗下的理财1号累计净值为0.5376元，今年来的收益为2.8%。 

        """
        self.model = model
        if self.model not in ["cnn", "rnn", "rnn_attention"]:
            raise Exception("model must in cnn, rnn, rnn_attention")
        self.document = [self.__select_num_words(context1), self.__select_num_words(context2)]
        self._currPath = os.path.dirname(__file__)
        self._vocabDict = self.__load_chinese_vocab()
        self.category_names = self.__label()

    @staticmethod
    def __select_num_words(cur):
        if len(cur) <= 1200:
            return cur
        return cur[0:600] + "。" + cur[len(cur)-600:]

    @property
    def data(self):
        if self.model == "cnn":
            deal_x = self.__deal_batch_data_2d(self.document)
            deal_x = self.__padding_batch_data_2d(deal_x, 1200)
            return np.array(deal_x)
        elif self.model == "rnn":
            deal_x = self.__deal_batch_data_2d(self.document)
            deal_x = self.__padding_batch_data_2d(deal_x)
            return np.array(deal_x)
        else:
            deal_x = self.__deal_batch_data_3d(self.document)
            deal_x = self.__padding_batch_data_3d(deal_x)
            return np.array(deal_x)

    @staticmethod
    def __label():
        tag = "体育,财经,房产,家居,教育,科技,时尚,时政,游戏,娱乐"
        category_names = dict()
        for i, v in enumerate(tag.split(",")):
            category_names[i] = v
        return category_names

    def __load_chinese_vocab(self):
        cv = dict()
        with codecs.open(os.path.join(self._currPath, "data/chinese_vocab.txt"), "r", "utf8") as f:
            for i, line in enumerate(f.readlines()):
                cv[line.strip()] = i
        return cv

    def __deal_batch_data_3d(self, document_lst):
        dataset_x = []
        for document in document_lst:
            sentence_lst = []
            for sentence in document.split("。"):
                char_lst = []
                if len(sentence) <= 3:
                    continue
                for char in sentence:
                    vocab_id = self._vocabDict.get(char, -1)
                    if vocab_id == -1:
                        continue
                    char_lst.append(vocab_id)
                if not char_lst:
                    continue
                sentence_lst.append(char_lst)
            dataset_x.append(sentence_lst)
        return dataset_x

    def __deal_batch_data_2d(self, document_lst):
        dataset_x = []
        for document in document_lst:
            char_lst = []
            for _char in document:
                vocab_id = self._vocabDict.get(_char, -1)
                if vocab_id == -1:
                    continue
                char_lst.append(vocab_id)
            dataset_x.append(char_lst)
        return dataset_x

    @staticmethod
    def __padding_batch_data_3d(deal_x):
        max_len_document = max([len(document) for document in deal_x])
        max_len_sentence = max(
            [max(_len) for _len in [[len(sentence) for sentence in document] for document in deal_x]])
        for document in deal_x:
            for sentence in document:
                sentence.extend((max_len_sentence - len(sentence)) * [0])
            document.extend((max_len_document - len(document)) * [max_len_sentence * [0]])
        return deal_x

    @staticmethod
    def __padding_batch_data_2d(deal_x, sequence_length=None):
        if sequence_length:
            max_len_document = max(max([len(document) for document in deal_x]), sequence_length)
        else:
            max_len_document = max([len(document) for document in deal_x])
        for document in deal_x:
            document.extend((max_len_document - len(document)) * [0])
        return deal_x


class PredictModel(object):
    def __init__(self):
        self.cnn_model_path = "model/cnn/"
        self.rnn_model_path = "model/rnn/"
        self.rnn_attention_model_path = "model/rnn_attention/"

    def __cnn_by_meta_graph(self):
        checkpoint_file = tf.train.latest_checkpoint(self.cnn_model_path)
        graph = tf.Graph()
        with graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.sess, checkpoint_file)
                self.inputs = graph.get_operation_by_name("input_x").outputs[0]
                self.keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
                self.predictions = graph.get_operation_by_name("score/predictions").outputs[0]

    def __rnn_by_meta_graph(self):
        checkpoint_file = tf.train.latest_checkpoint(self.rnn_model_path)
        graph = tf.Graph()
        with graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.sess, checkpoint_file)
                self.inputs = graph.get_operation_by_name("input_x").outputs[0]
                self.keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
                self.predictions = graph.get_operation_by_name("logits/predictions").outputs[0]

    def __rnn_attention_by_meta_graph(self):
        checkpoint_file = tf.train.latest_checkpoint(self.rnn_attention_model_path)
        graph = tf.Graph()
        with graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.sess, checkpoint_file)
                self.inputs = graph.get_operation_by_name("inputs").outputs[0]
                self.keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
                self.predictions = graph.get_operation_by_name("train_op/predictions").outputs[0]

    def cnn_predict_by_meta_graph(self, input_batch):
        self.__cnn_by_meta_graph()
        batch_predictions = self.sess.run(self.predictions, {self.inputs: input_batch, self.keep_prob: 1.0})
        return batch_predictions

    def rnn_predict_by_meta_graph(self, input_batch):
        self.__rnn_by_meta_graph()
        batch_predictions = self.sess.run(self.predictions, {self.inputs: input_batch, self.keep_prob: 1.0})
        return batch_predictions

    def rnn_attention_predict_by_meta_graph(self, input_batch):
        self.__rnn_attention_by_meta_graph()
        batch_predictions = self.sess.run(self.predictions, {self.inputs: input_batch, self.keep_prob: 1.0})
        return batch_predictions


if __name__ == "__main__":
    pm = PredictModel()
    #  predict by cnn model
    pd_cnn = PrepareData("cnn")
    preds = pm.cnn_predict_by_meta_graph(pd_cnn.data)
    print("采用cnn模型预测结果为:\t", [pd_cnn.category_names[pred] for pred in preds])

    #  predict by rnn model
    pd_rnn = PrepareData("rnn")
    preds = pm.rnn_predict_by_meta_graph(pd_rnn.data)
    print("采用rnn模型预测结果为:\t", [pd_rnn.category_names[pred] for pred in preds])

    # predict by rnn_attention model
    pd_rnn_attention = PrepareData("rnn_attention")
    preds = pm.rnn_attention_predict_by_meta_graph(pd_rnn_attention.data)
    print("采用rnn_attention模型预测结果为\t", [pd_rnn.category_names[pred] for pred in preds])
