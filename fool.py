from visdom import Visdom
import time
import csv
import argparse
import torch
import gc


from data import MyDataset
from vocab import Vocab
from Attacker import adversarial_paraphrase, textfool_perturb_text, random_attack
from preprocess import Tokenizer
from config import config_model_lists, config_dataset_list, \
    config_data, config_attack_list, config_dataset, config_device
from synonym import SynonymGenerator
from model_builder import build_TextCNN, build_LSTM_model
from tools import logging, parse_bool, str2seq, \
    get_time, make_dir_if_not_exist, \
    write_standard_data, read_standard_data


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='choose the dataset', choices=config_dataset_list, default=None)
parser.add_argument('--model', help='choose the target model',
                    choices=config_model_lists, default=None)
parser.add_argument('--attack', help='choose a attack algorithm', choices=config_attack_list, default=None)
parser.add_argument('--sub_rate_limit', type=float, default=None)
parser.add_argument('--note', type=str, default='')
parser.add_argument('--verbose', type=parse_bool, default='no')

args = parser.parse_args()
assert args.dataset == config_dataset
assert args.attack is not None

env_name = f'Fool_{args.attack}_{args.dataset}_{args.model}_subrate_{args.sub_rate_limit}'

if args.verbose:
    vis = Visdom(port=15555)
    assert vis.check_connection()
    opts = dict(
        title='rate',
        xlabel='nth',
        legend=['acc', 'att_failure_rate', 'att_success_rate']
    )
    vis.line([0], [0], win='rate', env=env_name, opts=opts)




output_dir = f'./static/{args.dataset}/foolresult/{args.attack}/{args.model}/'
make_dir_if_not_exist(output_dir)



dataset = args.dataset
attack_method = args.attack
sub_rate_limit = args.sub_rate_limit
csv_path = output_dir + args.note + '_' + get_time() + '_log.csv'
adv_path = output_dir + args.note + '_' + get_time() + '_adv.txt'
clean_path = config_data[dataset].clean_1k_path
model = args.model

tokenizer = Tokenizer()
train_paths = [config_data[dataset].train_data_path]
if 'adv' in model: train_paths.append(config_data[dataset].adv_train_path[model.split('_')[0]])
train_data = MyDataset(config_dataset, tokenizer, is_train=True, data_path=train_paths)
vocab0 = Vocab(train_data.data_token, vocab_limit_size=config_data[dataset].vocab_limit_size)


syn = SynonymGenerator()
syn_csv_path = config_data[dataset].syn_path
syn.read_syn_csv(syn_csv_path)
syn.build_word_index_map(vocab0)



class Fooler():

    def __init__(self, dataset:'MyDataset', sample_size_by_single:int, clean_samples_path:str):
        if dataset is None:
            self.datas, self.labels = read_standard_data(clean_samples_path)
        else:
            self.datas, self.labels = dataset.sample_by_labels(sample_size_by_single)
            write_standard_data(self.datas, self.labels, clean_samples_path)
        self.verbose = None
        self.use_typos = None
        self.adv_datas = None


    def generate_adversarial_samples(self, path, adv_method:str, verbose=False, use_typos=True,
                                     tokenizer=None, vocab=None, net=None, change_log_path=None,
                                     sub_rate_limit=None):
        assert len(self.datas) > 0
        self.adv_methods = {
            'PWWS': self.get_fool_sentence_pwws,
            'TEXTFOOL': self.get_fool_sentence_textfool,
            'RANDOM': self.get_fool_sentence_random,
        }
        assert adv_method in self.adv_methods
        self.adv_methods = self.adv_methods[adv_method]
        self.use_typos = use_typos
        self.verbose = verbose
        self.adv_datas = []
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.net = net
        self.log_path = change_log_path
        self.sub_rate_limit = sub_rate_limit
        success_num = 0
        failure_num = 0
        try_all = 0
        logging(f'generate adversarial samples {len(self.datas)} with {adv_method} to {path}')
            # assert net.training == False
        for idx, data in enumerate(self.datas):
            adv_s, flag, end = self.adv_methods(data, self.labels[idx], idx)
            self.adv_datas.append(str(adv_s))
            if flag == 1:
                success_num += 1
                try_all += 1
                logging(f'The {idx}th adv successfully crafted, '
                        f'success rate is {success_num/try_all:.5f}, cost {end:.2f} seconds')
            elif flag == 0:
                failure_num += 1
                try_all += 1
                logging(f'The {idx}th adv example failed crafted, '
                        f'fail rate is {failure_num/try_all:.5f}, cost {end:.2f} seconds')
            del adv_s
            del flag
            del end

            if (idx+1) % 100 == 0:
                write_standard_data(self.adv_datas, self.labels[idx - 99:idx + 1], path, 'a')
                self.adv_datas.clear()
                gc.collect()


            if args.verbose:
                now = idx + 1
                acc = 1 - success_num / now
                success_rate = success_num / try_all if try_all > 0 else 0.0
                failure_rate = 1 - success_rate
                vis.line(X=[now], Y=[acc], env=env_name, win='rate', name='acc', update='append')
                vis.line(X=[now], Y=[success_rate], env=env_name, win='rate', name='att_success_rate', update='append')
                vis.line(X=[now], Y=[failure_rate], env=env_name, win='rate', name='att_failure_rate', update='append')


        logging(f'try to generate adv_samples {try_all} '
                f'generate successfully {success_num}  '
                f'failed {failure_num}  '
                f'origin samples num is {len(self.datas)}')
        if len(self.adv_datas) > 0:
            write_standard_data(self.adv_datas, self.labels[len(self.adv_datas):], path, 'a')


    def get_fool_sentence_textfool(self, sentence:str, label:int, index:int):
        start = time.perf_counter()
        maxlen = config_data[config_dataset].padding_maxlen
        flag = -1
        end = -1
        vector = str2seq(sentence, self.vocab, self.tokenizer, maxlen).to(config_device)
        predict = self.net.predict_class(vector)[0]
        if predict == label:
            sentence, adv_y, sub_rate, NE_rate, change_tuple_list = textfool_perturb_text(
                sentence, self.net, self.vocab, self.tokenizer, maxlen, label, self.use_typos,
                verbose=self.verbose, sub_rate_limit=self.sub_rate_limit,
            )
            if adv_y != label:
                flag = 1
            else:
                flag = 0
            self.__write_log(index, flag, sub_rate, NE_rate, change_tuple_list)
            end = time.perf_counter() - start
        return sentence, flag, end

    def get_fool_sentence_random(self, sentence:str, label:int, index:int):
        start = time.perf_counter()
        maxlen = config_data[config_dataset].padding_maxlen
        vector = str2seq(sentence, self.vocab, self.tokenizer, maxlen).to(config_device)
        label = torch.tensor(label).to(config_device)
        flag = -1
        predict = self.net.predict_class(vector)[0]
        end = -1
        if predict == label:
            sentence, adv_y, sub_rate, NE_rate, change_tuple_list = random_attack(
                sentence, label, self.net, self.vocab, self.tokenizer, maxlen, self.verbose,
                self.sub_rate_limit
            )
            if adv_y != label:
                flag = 1
            else:
                flag = 0
            self.__write_log(index, flag, sub_rate, NE_rate, change_tuple_list)
            end = time.perf_counter() - start
        return sentence, flag, end

    def get_fool_sentence_pwws(self, sentence:str, label:int, index:int):
        start = time.perf_counter()
        maxlen = config_data[config_dataset].padding_maxlen
        vector = str2seq(sentence, self.vocab, self.tokenizer, maxlen).to(config_device)
        label = torch.tensor(label).to(config_device)
        flag = -1
        predict = self.net.predict_class(vector)[0]
        end = -1
        if predict == label:
            sentence, adv_y, sub_rate, NE_rate, change_tuple_list = adversarial_paraphrase(
                sentence, vector, label, self.tokenizer, self.vocab, self.net, self.verbose,
                self.sub_rate_limit,
            )
            if adv_y != label:
                flag = 1
            else:
                flag = 0
            self.__write_log(index, flag, sub_rate, NE_rate, change_tuple_list)
            end = time.perf_counter() - start
        return sentence, flag, end

    def __write_log(self, index, flag, sub_rate, NE_rate, change_tuple_list):
        with open(self.log_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([index, flag, sub_rate, NE_rate])
            writer.writerow(change_tuple_list)


def run_fooler(net, vocab):
    fooler = Fooler(None, 500, clean_path)
    fooler.generate_adversarial_samples(adv_path, attack_method, verbose=False, use_typos=False,
                                        tokenizer=tokenizer, vocab=vocab, net=net, change_log_path=csv_path,
                                        sub_rate_limit=sub_rate_limit)

def fool_LSTM_model(adv=False):
    net = build_LSTM_model(dataset, vocab0, config_device, syn=None, is_adv=adv)
    run_fooler(net, vocab0)

def fool_LSTM_Enhanced():
    net = build_LSTM_model(dataset, vocab0, config_device, syn=syn)
    run_fooler(net, vocab0)

def fool_TextCNN(adv=False):
    net = build_TextCNN(dataset, vocab0, config_device, syn=None, is_adv=adv)
    run_fooler(net, vocab0)

def fool_TextCNN_Enhanced():
    net = build_TextCNN(dataset, vocab0, config_device, syn=syn)
    run_fooler(net, vocab0)

def fool_BidLSTM(adv=False):
    net = build_LSTM_model(dataset, vocab0, config_device, is_bid=True, syn=None, is_adv=adv)
    run_fooler(net, vocab0)

def fool_BidLSTM_enhanced():
    net = build_LSTM_model(dataset, vocab0, config_device, is_bid=True, syn=syn)
    run_fooler(net, vocab0)





if __name__ == '__main__':

    if model == 'LSTM':
        fool_LSTM_model()
    elif model == 'LSTM_adv':
        fool_LSTM_model(adv=True)
    elif model == 'LSTM_enhanced':
        fool_LSTM_Enhanced()
    elif model == 'TextCNN':
        fool_TextCNN()
    elif model == 'TextCNN_adv':
        fool_TextCNN(adv=True)
    elif model == 'TextCNN_enhanced':
        fool_TextCNN_Enhanced()
    elif model == 'BidLSTM':
        fool_BidLSTM()
    elif model == 'BidLSTM_adv':
        fool_BidLSTM(adv=True)
    elif model == 'BidLSTM_enhanced':
        fool_BidLSTM_enhanced()

