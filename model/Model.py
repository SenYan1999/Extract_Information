import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class P_Model(nn.Module):
    def __init__(self, bert_path, bert_dim, n_class, drop_p):
        super(P_Model, self).__init__()

        self.bert_model = BertModel.from_pretrained(bert_path)
        self.fc = nn.Linear(bert_dim, n_class)
        self.drop_p = drop_p

    def forward(self, x, seg_id):
        x_mask = (x != 0).int()
        out = self.bert_model(x, token_type_ids = seg_id, attention_mask = x_mask)[0]

        out = F.dropout(out[:, 0, :].squeeze(), p=self.drop_p)
        out = self.fc(out)
        out = torch.sigmoid(out)

        return out

class NER_Model(nn.Module):
    def __init__(self, bert_path, bert_dim, n_class, drop_p):
        super(NER_Model, self).__init__()

        self.bert_model = BertModel.from_pretrained(bert_path)
        self.fc = nn.Linear(bert_dim, n_class)
        self.drop_p = drop_p

    def mask_softmax(self, x, x_mask):
        x_mask = x_mask.unsqueeze(dim=2).expand(x.shape).float()
        x = x + (1 - x_mask) * -1e30
        return F.log_softmax(x, dim=-1)

    def forward(self, x, seg_id):
        x_mask = (x != 0).int()
        out = self.bert_model(x, token_type_ids = seg_id, attention_mask = x_mask)[0]

        out = F.dropout(out, p=self.drop_p)
        out = self.fc(out)
        out = self.mask_softmax(out, x_mask)

        return out
