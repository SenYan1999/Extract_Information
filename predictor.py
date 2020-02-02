import torch
import os
import jsonlines
import csv

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from utils import logger
from tqdm import tqdm

def process_test_data(filename):
    texts = []
    with jsonlines.open(filename, 'r') as reader:
        for line in reader:
            texts.append(line['text'])
    return texts

class PredictPDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.X, self.SEG_ID = self.preprocess_text()

    def preprocess_text(self):
        X, SEG_ID = [], []
        for text in self.texts:
            tokens = ['[CLS]'] + self.tokenizer.tokenize(text) + ['[SEP]']
            x = [self.tokenizer.convert_tokens_to_ids(t) for t in tokens]

            if len(x) <= self.max_len:
                x = x + [0] * (self.max_len - len(x))
            else:
                x = x[: self.max_len]

            seg_id = [0] * len(x)
            X.append(x)
            SEG_ID.append(seg_id)

        X, SEG_ID = torch.LongTensor(X), torch.LongTensor(SEG_ID)

        return X, SEG_ID

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.SEG_ID[index]

class PredictNERDataset(Dataset):
    def __init__(self, x, logit):
        self.x = x
        self.logit = logit

        self.X, self.P, self.SEG_ID = self.reconstruct_data()

    def reconstruct_data(self):
        X, P, SEG_ID = [], [], []

        for x, logit in zip(self.x, self.logit):
            ps = torch.where(logit > 0.5)[0]

            for p in ps:
                p = [p] * x.shape[0]
                seg_id = [0] * x.shape[0]
                X.append(x.unsqueeze(dim=0))
                P.append(p)
                SEG_ID.append(seg_id)

        X = torch.cat(X, dim=0)
        P = torch.LongTensor(P)
        SEG_ID = torch.LongTensor(SEG_ID)

        return X, P, SEG_ID

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.P[index], self.SEG_ID[index]

class Predictor:
    def __init__(self, raw_file, out_file, p_model, ner_model, idx2pred, idx2ner, bert_path, max_len, batch_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.raw_file = raw_file
        self.out_file = out_file
        self.p_model = p_model.to(self.device).eval()
        self.ner_model = ner_model.to(self.device).eval()
        self.idx2pred = idx2pred
        self.idx2ner = idx2ner
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(bert_path, 'vocab.txt'))
        self.max_len = max_len
        self.batch_size = batch_size

    def predict_p(self):
        logger.info('Predict predicate...')

        text = process_test_data(self.raw_file)
        p_dataset = PredictPDataset(text, self.tokenizer, self.max_len)
        p_dataloader = DataLoader(p_dataset, batch_size=self.batch_size)

        X, Logits = [], []

        with torch.no_grad():
            for batch in tqdm(p_dataloader):
                x, seg_id = map(lambda i: i.to(self.device), batch)
                logits = self.p_model(x, seg_id)

                X.append(x)
                Logits.append(logits)

        X, Logits = torch.cat(X, dim=0), torch.cat(Logits, dim=0)

        logger.info('Finish predicting predicate.')

        return X, Logits

    def predict_ner(self, x, logit):
        logger.info('Predict NER...')

        ner_dataset = PredictNERDataset(x, logit)
        ner_dataloader = DataLoader(ner_dataset, batch_size=self.batch_size)

        X, P, NER = [], [], []

        with torch.no_grad():
            for batch in tqdm(ner_dataloader):
                x, p, seg_id = map(lambda i: i.to(self.device), batch)

                mask = (x != 0).int()
                out = self.ner_model(x, seg_id, p, mask, self.device)
                ner = self.ner_model.predict(out, mask)

                X.append(x)
                P.append(p[:, 0])
                NER.append(ner)

        X = torch.cat(X, dim=0)
        P = torch.cat(P, dim=0)
        NER = torch.cat(NER, dim=0)
        assert X.shape[0] == P.shape[0] == NER.shape[0]

        logger.info('Finish predicting NER.')

        return X, P, NER

    def convert_idx_text(self, X, P, NER, out_file, idx2pred, idx2ner):
        result = csv.writer(open(out_file, 'w'))
        result.writerow(['Text', 'Subject', 'Predicate', 'Object'])

        pbar = tqdm(total=X.shape[0])

        for x, p, ner in zip(X, P, NER):
            text = self.tokenizer.convert_ids_to_tokens(x, skip_special_tokens=True)
            text = self.tokenizer.convert_tokens_to_string(text)

            p = idx2pred[p.item()]

            s_begin, s_end, o_begin, o_end = None, None, None, None
            ner = [idx2ner[n.item()] for n in ner if n >= 0]
            for i, label in enumerate(ner):
                if label == 'B-SUB':
                    s_begin = i
                if label in ['I-SUB', 'B-SUB']: # note: s may contain 1 char
                    s_end = i # s_end will be the last one that label == 'I-SUB'
                if label == 'B-OBJ':
                    o_begin = i
                if label == ['I-OBJ', 'B-OBJ']: # note: o may contain 1 char
                    o_end = i # o_end will be the last one that label == 'I-OBJ'

            s = self.tokenizer.convert_ids_to_tokens(x[s_begin: s_end + 1]) if s_begin and s_end else None
            o = self.tokenizer.convert_ids_to_tokens(x[o_begin: o_end + 1]) if o_begin and o_end else None

            s = self.tokenizer.convert_tokens_to_string(s) if s else ''
            o = self.tokenizer.convert_tokens_to_string(o) if o else ''

            result.writerow([text, s, p, o])
            pbar.update(1)

    def predict(self):
        logger.info('Predict...')

        x, logit = self.predict_p()
        x, p, ner = self.predict_ner(x, logit)

        self.convert_idx_text(x, p, ner, self.out_file, self.idx2pred, self.idx2ner)

        logger.info('Finish predicting.')
        logger.info('Out file: %s' % self.out_file)
