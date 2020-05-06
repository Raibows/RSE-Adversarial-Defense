import argparse
import torch
from tqdm import tqdm
from torch.nn import Module
import csv
from config import config_dataset_list, config_data, \
    config_model_lists, config_device
from vocab import Vocab
from tools import logging, read_fool_log
from model_builder import build_LSTM_model, build_TextCNN
from preprocess import Tokenizer
from torch.utils.data import DataLoader
from data import MyDataset
from synonym import SynonymGenerator


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='choose the dataset', choices=config_dataset_list, default=None)
parser.add_argument('--models', help='choose the evaluate models', nargs='+', default=config_model_lists)
parser.add_argument('--adv_paths', default=None, nargs='+')
parser.add_argument('--save_path', default=None)
args = parser.parse_args()

assert args.adv_paths is not None
assert len(args.adv_paths) == len(args.models)
assert args.save_path is not None



dataset_name = args.dataset
models = args.models
maxlen = config_data[dataset_name].padding_maxlen
tokenizer = Tokenizer()
test_path = config_data[dataset_name].test_data_path
train_path = config_data[dataset_name].train_data_path
adv_paths = args.adv_paths



def evaluate(dataset:MyDataset, net:Module):
    assert dataset.is_train == False
    assert len(dataset) > 0
    dataset = DataLoader(dataset, batch_size=1000, shuffle=False)
    test_loss = 0.0
    net.eval()
    logging('starting evaluate!')
    with torch.no_grad():
        correct = 0
        total = 0
        for (X, label) in tqdm(dataset):
            X, label = X.to(config_device), label.to(config_device)
            logits = net(X)
            predict = logits.argmax(dim=1)
            correct += predict.eq(label).float().sum().item()
            total += X.size(0)

    correct /= total
    test_loss /= len(test_data)
    logging(f'evaluate done! acc {correct:.5f}, test average loss {test_loss:.5f}')
    return correct


def write_results_to_file(models, results, logs):
    with open(args.save_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['model_name', 'dataset', 'origin_acc', 'clean_acc', 'adv_acc', 'acc_shift', 'success_rate', 'sub_rate'])
        for model, res, log in zip(models, results, logs):
            print(f'{model} {dataset_name} testdata {len(test_data)} all acc is {res[0]:.5f}')
            print(f'{model} {dataset_name} cleandata {len(clean_data)} acc is {res[1]:.5f}')
            print(f'{model} {dataset_name} advdata {len(clean_data)} acc is {res[2]:.5f}')
            print(f'{model} {dataset_name} advdata {log[0]}, mean sub_rate {log[1]:.5f}, mean NE_rate {log[2]:.5f}')
            acc_shift = res[1] - res[2]
            success_rate = acc_shift / res[1]
            writer.writerow([model, dataset_name, res[0], res[1], res[2], acc_shift, success_rate, log[1]])
    logging(f'evaluate results saved in {args.save_path}')

if __name__ == '__main__':

    clean_path = config_data[dataset_name].clean_1k_path


    test_data = MyDataset(dataset_name, tokenizer, is_train=False, data_path=test_path)
    clean_data = MyDataset(dataset_name, tokenizer, is_train=False, data_path=clean_path)
    train_data = MyDataset(dataset_name, tokenizer, is_train=True, data_path=train_path)


    vocab0 = Vocab(train_data.data_token, vocab_limit_size=config_data[dataset_name].vocab_limit_size, is_using_pretrained=True)


    test_data.token2seq(vocab0, maxlen)
    clean_data.token2seq(vocab0, maxlen)

    syn = SynonymGenerator()
    syn_csv_path = config_data[dataset_name].syn_path
    syn.read_syn_csv(syn_csv_path)
    syn.build_word_index_map(vocab0)

    results = []
    logs = []


    for model, adv_path in zip(models, adv_paths):
        assert model in config_model_lists
        adv_log_path = adv_path[:-7] + 'log.csv'
        adv_data = MyDataset(dataset_name, tokenizer, is_train=False, data_path=adv_path)
        if 'adv' in model:
            paths = [train_path, config_data[dataset_name].adv_train_path[model.split('_')[0]]]
            train_data_with_adv = MyDataset(dataset_name, tokenizer, is_train=True, data_path=paths)
            vocab1 = Vocab(train_data_with_adv.data_token,
                           vocab_limit_size=config_data[dataset_name].vocab_limit_size,
                           is_using_pretrained=True)
            test_data.token2seq(vocab1, maxlen)
            clean_data.token2seq(vocab1, maxlen)
            adv_data.token2seq(vocab1, maxlen)
        else: adv_data.token2seq(vocab0, maxlen)

        if model == 'LSTM':
            net = build_LSTM_model(dataset_name, vocab0, config_device, is_bid=False, syn=None)
        elif model == 'LSTM_adv':
            net = build_LSTM_model(dataset_name, vocab1, config_device, is_bid=False, syn=None, is_adv=True)
        elif model == 'LSTM_enhanced':
            net = build_LSTM_model(dataset_name, vocab0, config_device, is_bid=False, syn=syn)
        elif model == 'TextCNN':
            net = build_TextCNN(dataset_name, vocab0, config_device, syn=None)
        elif model == 'TextCNN_adv':
            net = build_TextCNN(dataset_name, vocab1, config_device, syn=None, is_adv=True)
        elif model == 'TextCNN_enhanced':
            net = build_TextCNN(dataset_name, vocab0, config_device, syn=syn)
        elif model == 'BidLSTM':
            net = build_LSTM_model(dataset_name, vocab0, config_device, is_bid=True, syn=None)
        elif model == 'BidLSTM_adv':
            net = build_LSTM_model(dataset_name, vocab1, config_device, is_bid=True, syn=None, is_adv=True)
        elif model == 'BidLSTM_enhanced':
            net = build_LSTM_model(dataset_name, vocab0, config_device, is_bid=True, syn=syn)

        logging('evaluate origin test data')
        test_acc = evaluate(test_data, net)
        logging('evaluate origin 1k clean data')
        clean1k_acc = evaluate(clean_data, net)
        logging('evaluate adversarial 1k data')
        adv1k_acc = evaluate(adv_data, net)

        results.append((test_acc, clean1k_acc, adv1k_acc))
        logs.append(read_fool_log(adv_log_path))

    write_results_to_file(models, results, logs)





