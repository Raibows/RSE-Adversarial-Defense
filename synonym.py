import spacy
from Attacker import _generate_synonym_candidates
from tools import logging, get_random
from tqdm import tqdm
import csv
import torch
import random
random.seed(667)


class SynonymGenerator():

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.pos_tag = {
            'ADJ': 'a',
            'ADV': 'r',
            'NOUN': 'n',
            'VERB': 'v',
            'ADP': 'v',
        }
        self.syn_dict = {}
        self.syn_count = 0
        self.syn_index_dict = {}

    def get_similarity_words(self, word:str) -> {str}:
        word = self.nlp(word)[0]
        return {res.candidate_word.lower() for res in _generate_synonym_candidates(word, -1)}

    def most_similar(self, word):
        word = self.nlp(word)[0]
        queries = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
        by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
        return by_similarity[:10]

    def get_similarity(self, word_a:str, word_b:str):
        word_a = self.nlp(word_a)
        word_b = self.nlp(word_b)
        return float(word_a.similarity(word_b))

    def build_syn_dict(self, vocab:'Vocab', path):
        assert len(vocab.word_dict) > 0
        for key, value in tqdm(vocab.word_dict.items()):
            if value == 0: continue
            res = self.get_similarity_words(key)
            if len(res) > 0: self.syn_dict[key] = res
            self.syn_count += len(res)

        num = len(self.syn_dict)
        logging(f'synonymous words has been built, key word is {num}, synonymous words is {self.syn_count}')
        self.write_syn_csv(path)

    def read_syn_csv(self, path):
        with open(path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            for line in reader:
                res = set(line[1:])
                self.syn_count += len(res)
                self.syn_dict[line[0]] = res

        logging(f'load syn_words from {path}, key word is {len(self.syn_dict)}, synonymous words is {self.syn_count}')

    def write_syn_csv(self, path):
        with open(path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=',')
            for key, value in self.syn_dict.items():
                writer.writerow([key]+list(value))

        logging(f'write synoymous words {self.syn_count} to {path}')


    def build_word_index_map(self, vocab):
        assert len(self.syn_dict) > 0
        assert len(vocab.word_dict) > 0
        self.syn_index_dict[0] = set()
        for key, value in self.syn_dict.items():
            res = {vocab.get_index(v) for v in value}
            if get_random(0, 1) == 1:
                res.add(0)
            idx = vocab.get_index(key)
            if idx == 0:
                self.syn_index_dict[idx] |= res
            else:
                self.syn_index_dict[idx] = res
        unknowns = set(random.sample(self.syn_index_dict.keys(), 10))
        unknowns.add(0)
        self.syn_index_dict[0] |= set(unknowns)


    def get_syn_words(self, word: str) -> {str}:
        res = self.syn_dict.get(word)
        return {} if res is None else res

    def get_syn_words_index(self, word_index: int) -> {int}:
        res = self.syn_index_dict.get(word_index)
        return {} if res is None else res

    def random_mask(self, X:torch.Tensor, mask_low=1, mask_rate=0.15):
        limit = int(X.size()[-1] * mask_rate)
        limit = get_random(mask_low, limit)
        temp = [i for i in range(X.size()[-1])]
        count = 0
        replaced = set()
        loop = 0

        while count < limit and loop < 5:
            loop += 1
            pos = random.sample(temp, limit-count)
            for p in pos:
                if count == limit: break
                if p in replaced: continue
                ori_word = X[p].item()
                res = self.get_syn_words_index(ori_word)
                if len(res) > 0:
                    s = random.sample(res, 1)
                    X[p] = torch.tensor(s)
                    count += 1
                    # print(f'{count} sub with {p} {s}')
                    replaced.add(p)
        flag = True if count > 0 else False
        return X, flag








if __name__ == '__main__':
    import argparse
    from config import config_dataset_list, config_data
    from data import MyDataset
    from vocab import Vocab
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=config_dataset_list)
    args = parser.parse_args()
    dataset_name = args.dataset

    test = SynonymGenerator()
    train_data = MyDataset(dataset_name, None, is_train=True,
                           data_path=config_data[dataset_name].train_data_path)
    vocab = Vocab(train_data.data_token, is_using_pretrained=False,
                  vocab_limit_size=config_data[dataset_name].vocab_limit_size)
    syn_path = config_data[dataset_name].syn_path
    test.build_syn_dict(vocab, syn_path)

