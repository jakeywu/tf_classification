import os
import numpy as np
import codecs
import random


class PrepareClassifyData(object):
    def __init__(self, conf, mode="train"):
        self._currPath = os.path.dirname(__file__)
        self._config = conf
        self._mode = mode
        self._sourceData = self.__read_dataset()
        self._vocabDict = self.__load_chinese_vocab()
        self._categoryId = self.__classify_names()

    def __load_chinese_vocab(self):
        cv = dict()
        with codecs.open(os.path.join(self._currPath, "data/chinese_vocab.txt"), "r", "utf8") as f:
            for i, line in enumerate(f.readlines()):
                cv[line.strip()] = i
        return cv

    def __read_dataset(self):
        if self._mode == "train":
            dataset_path = os.path.join(self._currPath, "data/trainset.txt")
        elif self._mode == "test":
            dataset_path = os.path.join(self._currPath, "data/testset.txt")
        else:
            raise Exception("mode must be in [train/test]")
        if not os.path.exists(dataset_path):
            raise Exception("path [{}] not exists".format(dataset_path))

        with codecs.open(dataset_path, "r", "utf8") as fp:
            dataset = fp.readlines()
            random.shuffle(dataset)
            return iter(dataset)

    def __classify_names(self):
        category_id = dict()
        for i, v in enumerate(self._config.classify_names.split(",")):
            category_id[v] = i
        return category_id

    def __deal_batch_data(self, document_lst):
        dataset_x = []
        dataset_y = []
        for document in document_lst:
            _y = document.split("\t")[0]
            _x = document.lstrip(_y + "\t").strip()
            category_id = self._categoryId.get(_y, -1)
            if category_id == -1:
                continue

            char_lst = []
            for _char in _x:
                vocab_id = self._vocabDict.get(_char, -1)
                if vocab_id == -1:
                    continue
                char_lst.append(vocab_id)
                if not char_lst:
                    continue

            if not char_lst or not _y:
                continue
            dataset_x.append(char_lst)
            dataset_y.append(category_id)
        return dataset_x, dataset_y

    @staticmethod
    def __padding_batch_data(deal_x):
        max_len_document = max([len(document) for document in deal_x])
        for document in deal_x:
            document.extend((max_len_document - len(document)) * [0])
        return deal_x

    def __next__(self):
        document_lst = []
        count = 0
        try:
            while count < self._config.batch_size:
                cur = next(self._sourceData)
                if not cur:
                    continue
                count += 1
                document_lst.append(cur[0:600])
        except StopIteration as iter_exception:
            if count == 0:
                raise iter_exception

        deal_x, deal_y = self.__deal_batch_data(document_lst)
        deal_x = self.__padding_batch_data(deal_x)

        return np.array(deal_x, dtype=np.int32), np.array(deal_y, dtype=np.int32)

    def __iter__(self):
        return self

