from config import config_data
from torch.utils.data import Dataset
import torch
from tools import logging, read_standard_data
import random
from preprocess import Tokenizer

class MyDataset(Dataset):

    def __init__(self, dataset_name, tokenizer, is_train:bool, data_path, read_data_func=None, is_to_tokens=True):
        self.is_train = is_train
        self.dataset_name = dataset_name
        self.labels_num = config_data[dataset_name].labels_num
        self.data = []
        self.labels = []
        self.data_token = []
        self.data_seq = []
        self.labels_tensor = []
        self.vocab = None
        self.tokenizer = tokenizer if tokenizer else Tokenizer('normal', remove_stop_words=False)
        self.maxlen = None
        if isinstance(data_path, str):
            data_path = [data_path]
        for path in data_path:
            if read_data_func is not None: td, tl = read_data_func(path)
            else: td, tl = read_standard_data(path)
            self.data += td
            self.labels += tl

        if is_to_tokens:
            self.data2token()


    def data2token(self):
        logging(f'data is train {self.is_train} is to tokens!')
        assert self.data is not None
        for sen in self.data:
            self.data_token.append(self.tokenizer(sen))

    def token2seq(self, vocab:'Vocab', maxlen:int):
        if len(self.data_seq) > 0:
            self.data_seq.clear()
            self.labels_tensor.clear()
        logging(f'data is train {self.is_train} is to sequence!')
        self.vocab = vocab
        self.maxlen = maxlen
        assert self.data_token is not None
        for tokens in self.data_token:
            self.data_seq.append(self.__encode_tokens(tokens))
        for label in self.labels:
            self.labels_tensor.append(torch.tensor(label))

    def __encode_tokens(self, tokens)->torch.Tensor:
        '''
        if one sentence length is shorter than maxlen, it will use pad word for padding to maxlen
        :param tokens:
        :return:
        '''
        pad_word = 0
        x = [pad_word for _ in range(self.maxlen)]
        temp = tokens[:self.maxlen]
        for idx, word in enumerate(temp):
            x[idx] = self.vocab.get_index(word)
        return torch.tensor(x)

    def split_data_by_label(self):
        datas = [[] for _ in range(self.labels_num)]
        for idx, lab in enumerate(self.labels):
            temp = (self.data[idx], lab)
            datas[lab].append(temp)
        return datas

    def sample_by_labels(self, single_label_num:int):
        datas = self.split_data_by_label()
        sample_data = []
        sample_label = [-1 for _ in range(single_label_num*self.labels_num)]
        for i in range(self.labels_num):
            sample_data += random.sample(datas[i], single_label_num)
        for idx, data in enumerate(sample_data):
            sample_data[idx] = data[0]
            sample_label[idx] = data[1]
        assert len(sample_data) == len(sample_label)
        return sample_data, sample_label


    def statistic(self):
        import numpy as np
        length = [len(x) for x in self.data_token]
        print(f'statistic {self.dataset_name} isTrain {self.is_train}, '
              f'maxlen {max(length)}, minlen {min(length)}, '
              f'meanlen {sum(length) / len(length)}, medianlen {np.median(length)}')

    def __len__(self):
        return len(self.labels_tensor)

    def __getitem__(self, item):
        return (self.data_seq[item], self.labels_tensor[item])





if __name__ == '__main__':
    pass

