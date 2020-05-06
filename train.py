import re
import argparse
import copy
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from visdom import Visdom
from vocab import Vocab
from preprocess import Tokenizer
from data import MyDataset
from synonym import SynonymGenerator
from model_builder import build_LSTM_model, build_TextCNN
from config import config_dataset_list, config_model_lists, \
    config_data, config_model_load_path, config_device, \
    config_model_save_path
from tools import parse_bool, logging, get_time



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=config_dataset_list)
parser.add_argument('--model', choices=config_model_lists)
# parser.add_argument('--save_acc_limit', help='set a acc lower limit for saving model',
#                     type=float, default=0.85)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--enhanced', type=parse_bool, choices=[True, False])
parser.add_argument('--adv', choices=[True, False], default='no', type=parse_bool)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--note', type=str, default='')
parser.add_argument('--load_model', choices=[True, False], default='no', type=parse_bool)
parser.add_argument('--verbose', choices=[True, False], default='no', type=parse_bool)
args = parser.parse_args()




dataset_name = args.dataset
# save_acc_limit = args.save_acc_limit
maxlen = config_data[dataset_name].padding_maxlen
epoch = args.epoch
batch = args.batch
lr = args.lr
note = args.note
is_load_model = args.load_model
model_name = args.model
is_enhanced = args.enhanced

env_name = dataset_name + '_' + model_name + '_enhanced_' + str(is_enhanced)

if args.verbose:
    vis = Visdom(port=15555)
    assert vis.check_connection()
    opts = dict(
        title='loss',
        xlabel='epoch',
        legend=['train_loss', 'test_loss']
    )
    vis.line([0], [0], win='loss', env=env_name, opts=opts)
    vis.line([0], [0], win='acc', env=env_name, opts=dict(title='acc', legend=['test_acc']))

tokenizer = Tokenizer()
train_paths = [config_data[dataset_name].train_data_path]
if args.adv: train_paths.append(config_data[dataset_name].adv_train_path[model_name])
train_data = MyDataset(dataset_name, tokenizer, is_train=True, data_path=train_paths)
test_data = MyDataset(dataset_name, tokenizer, is_train=False, data_path=config_data[dataset_name].test_data_path)

vocab = Vocab(train_data.data_token, vocab_limit_size=config_data[dataset_name].vocab_limit_size)
train_data.token2seq(vocab, maxlen, False)
test_data.token2seq(vocab, maxlen, False)
if is_enhanced:
    syn = SynonymGenerator()
    syn_csv_path = config_data[dataset_name].syn_path
    syn.read_syn_csv(syn_csv_path)
    syn.build_word_index_map(vocab)
else: syn = None

test_data = DataLoader(test_data, batch_size=batch)
train_data = DataLoader(train_data, batch_size=batch, shuffle=True)



if model_name == 'LSTM':
    net = build_LSTM_model(dataset_name, vocab, config_device, syn=syn, is_load=is_load_model, is_adv=args.adv)
elif model_name == 'TextCNN':
    net = build_TextCNN(dataset_name, vocab, config_device, syn=syn, is_load=is_load_model, is_adv=args.adv)
elif model_name == 'BidLSTM':
    net = build_LSTM_model(dataset_name, vocab, config_device, is_bid=True, syn=syn, is_load=is_load_model, is_adv=args.adv)


# net = nn.DataParallel(net, device_ids=[0, 1])
net.to(config_device)
best_path = config_model_load_path[dataset_name].get(net.model_name)
best_state = None
best_acc = 0.0 if is_load_model == False else float(re.findall("_\d.\d+_", best_path)[0][1:-1])
if is_load_model: logging(f'loading net model from {best_path}, acc is {best_acc}')
else: pass # net.apply(weights_init)


def main(epochs: int, learning_rate: float):
    global best_acc
    global best_path
    global note
    global best_state

    criterion = nn.CrossEntropyLoss().to(config_device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=3,
                                                     verbose=True, min_lr=3e-9)
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: 1e-2 if ep < 3 else 1)


    loss = 0.0
    train_loss = 0.0

    global_batch_idx = 1
    acc_all = []
    test_best_loss = 1e5

    def evaluate(now_ep)->float:
        test_loss = 0.0
        logging('starting evaluate!')
        net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for (X, label) in tqdm(test_data):
                X, label = X.to(config_device), label.to(config_device)
                logits = net(X)
                test_loss += criterion(logits, label).item()
                predict = logits.argmax(dim=1)
                correct += predict.eq(label).float().sum().item()
                total += X.size(0)

        correct /= total
        test_loss /=  len(test_data)
        logging(f'epoch {now_ep} evaluate done! test acc {correct:.5f}, best acc{best_acc:.5f}, test batch loss {test_loss:.5f}')
        return correct, test_loss

    for ep in range(epochs):
        logging(f'epoch {ep} start!')
        net.train()
        train_loss = 0.0
        for batch_idx, (X, label) in enumerate(tqdm(train_data)):
            X, label = X.to(config_device), label.to(config_device)
            logits = net(X)
            loss = criterion(logits, label)

            train_loss += loss.item()
            global_batch_idx += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if global_batch_idx % 2000 == 0:
                logging(f'global batch {global_batch_idx}, loss is {train_loss/(batch_idx+1):.5f}')

        train_loss /= len(train_data)
        logging(f'epoch {ep} train done! averge train loss is {train_loss:.5f}')


        ep_test_acc, test_loss = evaluate(ep)
        acc_all.append(ep_test_acc)
        if ep < 4: warmup_scheduler.step(ep)
        else: scheduler.step(test_loss, epoch=ep)

        if args.verbose:
            vis.line(X=[ep + 1], Y=[train_loss], env=env_name, win='loss', name='train_loss', update='append')
            vis.line(X=[ep + 1], Y=[test_loss], env=env_name, win='loss', name='test_loss', update='append')
            vis.line(X=[ep + 1], Y=[ep_test_acc], env=env_name, win='acc', name='test_acc', update='append')

        if ep_test_acc > best_acc or (ep_test_acc == best_acc and test_loss < test_best_loss):
            best_acc = ep_test_acc
            test_best_loss = test_loss
            best_path = config_model_save_path[dataset_name].format(net.model_name, best_acc, get_time(), note)
            best_state = copy.deepcopy(net.state_dict())

        if (ep+1) % (epochs // 3) == 0 and best_state:
            logging(f'saving model in {best_path} best acc {best_acc:.5f}')
            torch.save(best_state, best_path)
            best_state = None


    count = epochs - epochs // 2
    acc_all = sum(acc_all[epochs // 2:])
    acc_all /= count
    if best_state is not None: torch.save(best_state, best_path)
    logging(f'train {epochs} done! The last train loss is {train_loss/len(test_data):.5f}!')
    logging(f'The last {count} epoch test average acc is {acc_all:.5f}, best acc is {best_acc:.5f}\n'
            f'best model saved in {best_path}')







if __name__ == '__main__':
    main(epochs=epoch, learning_rate=lr)
