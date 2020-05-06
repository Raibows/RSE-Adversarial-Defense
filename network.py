import torch
from torch import nn
from vocab import Vocab
from torch.nn import functional as F
from config import config_RSE_mask_low, config_RSE_mask_rate


class LSTMModel(nn.Module):

    def __init__(self,
                 num_hiddens:int, num_layers:int, word_dim:int, vocab:Vocab, labels:int,
                 using_pretrained=True, bid=False, head_tail=False, syn=None, adv=False):
        super(LSTMModel, self).__init__()
        if bid:
            self.model_name = 'BidLSTM' if syn is None else 'BidLSTM_enhanced'
        else:
            self.model_name = 'LSTM' if syn is None else 'LSTM_enhanced'
        if adv:
            assert syn is None
            self.model_name += '_adv'
        self.head_tail = head_tail
        self.bid = bid
        self.syn = syn

        self.embedding_layer = nn.Embedding(vocab.num, word_dim)
        self.embedding_layer.weight.requires_grad = True
        if using_pretrained:
            assert vocab.word_dim == word_dim
            assert vocab.num == vocab.vectors.shape[0]
            self.embedding_layer.from_pretrained(torch.from_numpy(vocab.vectors))
            self.embedding_layer.weight.requires_grad = False

        self.dropout = nn.Dropout(0.5)

        self.encoder = nn.LSTM(
            input_size=word_dim, hidden_size=num_hiddens,
            num_layers=num_layers, bidirectional=bid,
            dropout=0.3
        )

        # using bidrectional, *2
        if bid:
            num_hiddens *= 2
        if head_tail:
            num_hiddens *= 2


        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_hiddens, labels),
        )


    def __random_mask(self, X):
        for idx, one in enumerate(X):
            X[idx], flag = self.syn.random_mask(one, mask_low=config_RSE_mask_low, mask_rate=config_RSE_mask_rate)
        return X

    def forward(self, X: torch.Tensor, y=None):
        if self.syn:
            X = self.__random_mask(X)

        X = X.permute(1, 0) # [batch, seq_len] -> [seq_len, batch]
        X = self.embedding_layer(X)  #[seq_len, batch, word_dim]

        X = self.dropout(X)

        outputs, _ = self.encoder(X) # output, (hidden, memory)
        # outputs [seq_len, batch, hidden*2] *2 means using bidrectional
        # head and tail, [batch, hidden*4]

        outputs = torch.cat((outputs[0], outputs[-1]), -1) if self.head_tail else outputs[-1]

        outputs = self.fc(outputs) # [batch, hidden*4] -> [batch, labels]

        return outputs

    def predict_prob(self, X: torch.Tensor, y_true: torch.Tensor)->[float]:
        if self.training:
            raise RuntimeError('you shall take the model in eval to get probability!')
        if X.dim() == 1:
            X = X.view(1, -1)
        if y_true.dim() == 0:
            y_true = y_true.view(1)

        with torch.no_grad():
            logits = self(X)
            logits = F.softmax(logits, dim=1)
            prob = [logits[i][y_true[i]].item() for i in range(y_true.size(0))]
            return prob

    def predict_class(self, X:torch.Tensor)->[int]:
        if self.training:
            raise RuntimeError('you shall take the model in eval to get probability!')
        if X.dim() == 1:
            X = X.view(1, -1)
        predicts = None
        with torch.no_grad():
            logits = self(X)
            logits = F.softmax(logits, dim=1)
            predicts = [one.argmax(0).item() for one in logits]
        return predicts



class TextCNN(nn.Module):
    def __init__(self, vocab:Vocab, train_embedding_word_dim, is_static, using_pretrained,
                 num_channels:list, kernel_sizes:list, labels:int, syn=None, adv=False):
        super(TextCNN, self).__init__()
        self.model_name = 'TextCNN' if syn is None else 'TextCNN_enhanced'
        if adv:
            assert syn is None
            self.model_name += '_adv'

        self.syn = syn
        self.using_pretrained = using_pretrained
        self.word_dim = train_embedding_word_dim
        if using_pretrained: self.word_dim += vocab.word_dim

        if using_pretrained:
            self.embedding_pre = nn.Embedding(vocab.num, vocab.word_dim)
            self.embedding_pre.from_pretrained(torch.from_numpy(vocab.vectors))
            self.embedding_pre.weight.requires_grad = not is_static

        self.embedding_train = nn.Embedding(vocab.num, train_embedding_word_dim)

        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(
                nn.Conv1d(in_channels=self.word_dim,
                          out_channels=c,
                          kernel_size=k)
            )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(sum(num_channels), labels)


    def __random_mask(self, X):
        for idx, one in enumerate(X):
            X[idx], flag = self.syn.random_mask(one, mask_low=config_RSE_mask_low, mask_rate=config_RSE_mask_rate)
        return X

    def forward(self, X:torch.Tensor):
        if self.syn:
            X = self.__random_mask(X)
        if self.using_pretrained:
            embeddings = torch.cat((
                self.embedding_train(X),
                self.embedding_pre(X),
            ), dim=-1) # [batch, seqlen, word-dim0 + word-dim1]
        else: embeddings = self.embedding_train(X)

        embeddings = self.dropout(embeddings)

        embeddings = embeddings.permute(0, 2, 1) # [batch, dims, seqlen]


        outs = torch.cat(
            [self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)

        outs = self.dropout(outs)

        logits = self.fc(outs)
        return logits

    def predict_prob(self, X: torch.Tensor, y_true: torch.Tensor) -> [float]:
        if self.training:
            raise RuntimeError('you shall take the model in eval to get probability!')
        if X.dim() == 1:
            X = X.view(1, -1)
        if y_true.dim() == 0:
            y_true = y_true.view(1)

        with torch.no_grad():
            logits = self(X)
            logits = F.softmax(logits, dim=1)
            prob = [logits[i][y_true[i]].item() for i in range(y_true.size(0))]
            return prob

    def predict_class(self, X: torch.Tensor) -> [int]:
        if self.training:
            raise RuntimeError('you shall take the model in eval to get probability!')
        if X.dim() == 1:
            X = X.view(1, -1)
        predicts = None
        with torch.no_grad():
            logits = self(X)
            logits = F.softmax(logits, dim=1)
            predicts = [one.argmax(0).item() for one in logits]
        return predicts



def weights_init(m):
    if isinstance(m, nn.Linear):
        initrange = 0.5
        m.weight.data.uniform_(-initrange, initrange)
        m.bias.data.zero_()



if __name__ == '__main__':
    pass