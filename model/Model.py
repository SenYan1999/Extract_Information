import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import numpy as np
from model.transformer import *
from args import Hyperparams as hp
from model.crf import CRF

class P_Model(nn.Module):
    def __init__(self, bert_path, bert_dim, n_class, drop_p, max_len):
        super(P_Model, self).__init__()

        self.bert_model = BertModel.from_pretrained(bert_path)
        self.fc = nn.Linear(bert_dim, n_class)
        self.maxpool = nn.MaxPool1d(max_len)
        self.drop_p = drop_p

    def forward(self, x, seg_id):
        x_mask = (x != 0).int()
        out = self.bert_model(x, token_type_ids=seg_id, attention_mask=x_mask)[0]
        out = F.relu(out)
        out = self.maxpool(out.permute(0, 2, 1)).squeeze()

        out = F.dropout(out, p=self.drop_p)
        out = self.fc(out)
        out = torch.sigmoid(out)

        return out

class NER_Model(nn.Module):
    def __init__(self, bert_path, bert_dim, n_class, drop_p,num_pre, max_len):
        super(NER_Model, self).__init__()

        self.bert_model = BertModel.from_pretrained(bert_path)
        self.fc = nn.Linear(bert_dim*2, n_class)
        self.dropout = nn.Dropout(drop_p)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # pre embedding
        self.pre_dim = bert_dim
        self.pre_embedding = nn.Embedding(num_pre, self.pre_dim)
        self.pre_embedding.weight.data.copy_(torch.from_numpy(
            self.random_embedding_label(num_pre, self.pre_dim, 0.025)))

        # transformer
        self.enc_positional_encoding = positional_encoding(768, zeros_pad=True, scale=True)
        for i in range(hp.num_blocks):
            self.__setattr__('enc_self_attention_%d' % i, multihead_attention(num_units=hp.hidden_units,
                                                                              num_heads=hp.num_heads,
                                                                              dropout_rate=hp.dropout_rate,
                                                                              causality=False))
            self.__setattr__('enc_feed_forward_%d' % i, feedforward(hp.hidden_units,
                                                                    [4 * hp.hidden_units,
                                                                     hp.hidden_units]))


        # crf
        self.crf = CRF(n_class, use_cuda=True if torch.cuda.is_available() else False)


    def random_embedding_label(self, vocab_size, embedding_dim, scale):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        # scale = np.sqrt(3.0 / embedding_dim)
        # scale = 0.025
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def encoder(self, embed, device):
        # Dropout
        self.enc = self.dropout(embed)
        # Blocks
        for i in range(hp.num_blocks):
            self.enc = self.__getattr__('enc_self_attention_%d' % i)(self.enc, self.enc, self.enc, device)
            # Feed Forward
            self.enc = self.__getattr__('enc_feed_forward_%d' % i)(self.enc)  # q和k一样，所以叫自注意力机制 (b, l, 768)
        return self.enc

    def forward(self, x, seg_id, p, mask, device):
        self.input_ids=x
        x_mask = (x != 0).int()
        out = self.bert_model(x, token_type_ids = seg_id, attention_mask = x_mask)[0]
        # Positional Encoding (b, l, 768)
        pos = self.enc_positional_encoding(self.input_ids)
        out += pos.to(self.device)
        # word embedding + pre_embedding
        pre_embedding=self.pre_embedding(p)
        pre_embedding=pre_embedding*mask.unsqueeze(dim=2).expand(mask.size()[0],mask.size()[1], self.pre_dim)
        out=torch.cat((out,pre_embedding), 2)
        # transformer output
        encoder_out = self.encoder(out,device)
        out=self.fc(encoder_out)

        return out

    def loss_fn(self, transformer_encode, output_mask, tags):
        loss = self.crf.negative_log_loss(transformer_encode, output_mask, tags)
        return loss.cpu()

    def predict(self, transformer_encode, output_mask):
        predicts = self.crf.get_batch_best_path(transformer_encode, output_mask)  # (b,l)
        return predicts

