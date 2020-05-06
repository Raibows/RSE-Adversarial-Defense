import torch
from network import LSTMModel, TextCNN
from config import LSTMConfig, TextCNNConfig, \
    config_data, config_model_load_path




def build_LSTM_model(dataset, vocab, dev, is_bid=False, syn=None, is_adv=False, is_load=True):

    num_hiddens = LSTMConfig.num_hiddens[dataset]
    num_layers = LSTMConfig.num_layers[dataset]
    is_using_pretrained = LSTMConfig.is_using_pretrained[dataset]
    word_dim = LSTMConfig.word_dim[dataset]
    net = LSTMModel(num_hiddens=num_hiddens, num_layers=num_layers, word_dim=word_dim, bid=is_bid,
                    head_tail=True, vocab=vocab, labels=config_data[dataset].labels_num, syn=syn,
                    using_pretrained=is_using_pretrained, adv=is_adv)
    if is_load:
        model_path = config_model_load_path[dataset][net.model_name]
        net.load_state_dict(torch.load(model_path, map_location=dev))
    net.to(dev)
    net.eval()
    return net



def build_TextCNN(dataset, vocab, dev, syn=None, is_adv=False, is_load=True):
    channel_size, kernel_size = TextCNNConfig.channel_kernel_size[dataset]
    is_static = TextCNNConfig.is_static[dataset]
    train_embedding_dim = TextCNNConfig.train_embedding_dim[dataset]
    net = TextCNN(vocab, train_embedding_dim, is_static, TextCNNConfig.using_pretrained[dataset],
                  channel_size, kernel_size, config_data[dataset].labels_num, syn=syn, adv=is_adv)
    if is_load:
        model_path = config_model_load_path[dataset][net.model_name]
        net.load_state_dict(torch.load(model_path, map_location=dev))
    net.to(dev)
    net.eval()
    return net
