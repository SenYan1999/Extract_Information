import jsonlines
import torch
import os
import logging
from logging import handlers
import sys
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer
from args import parser

# define global logging
args = parser.parse_args()


def init_logger(filename, when='D', backCount=3,
                fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
    logger = logging.getLogger(filename)
    format_str = logging.Formatter(fmt)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
    th.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(th)

    return logger


logger = init_logger(filename=args.log_file)


class KG_DataProcessor:
    def __init__(self, data_filename: str, count: int, predicate2idx: dict):
        self.data_filename = data_filename
        self.count = count

    def get_data(self):
        data = []
        with jsonlines.open(self.data_filename) as reader:
            for line in reader:
                data_piece = {}
                predicates = []
                subjects = []
                objects = []

                text = line['text'].lower()
                for spo in line['spo_list']:
                    predicates.append(spo['predicate'])
                    subjects.append(spo['subject'].lower())
                    objects.append(spo['object'].lower())

                data_piece['Text'] = text
                data_piece['Predicate'] = predicates
                data_piece['Subject'] = subjects
                data_piece['Object'] = objects

                data.append(data_piece)

        return data


class PDataset(Dataset):
    def __init__(self, data: list, max_len: int, bert_type: str, pred2idx):
        self.raw_data = data
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(bert_type, 'vocab.txt'))
        self.pred2idx = pred2idx

        self.X, self.SEG_ID, self.Y = self.preprocess_raw_data()

    def __getitem__(self, idx):
        return self.X[idx], self.SEG_ID[idx], self.Y[idx]

    def __len__(self):
        return self.Y.shape[0]

    def preprocess_raw_data(self):
        X, SEG_ID, Y = [], [], []

        for line in self.raw_data:
            text = line['Text']
            tokens = ['[CLS]'] + self.tokenizer.tokenize(text)
            x = [self.tokenizer.convert_tokens_to_ids(t) for t in tokens]

            if len(x) <= self.max_len:
                x = x + [self.tokenizer.convert_tokens_to_ids('[PAD]')] * (self.max_len - len(x))
            else:
                x = x[: self.max_len]

            predicate = line['Predicate']
            y = [0] * len(self.pred2idx)
            for p in predicate:
                y[self.pred2idx[p]] = 1
            seg_id = [0] * len(x)

            X.append(x)
            Y.append(y)
            SEG_ID.append(seg_id)

        X, SEG_ID, Y = torch.LongTensor(X), torch.LongTensor(SEG_ID), torch.LongTensor(Y)

        return X, SEG_ID, Y


class NERDataset(Dataset):
    def __init__(self, data: list, max_len: int, bert_type: str, pred2idx):
        self.raw_data = data
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(bert_type, 'vocab.txt'))
        self.pred2idx = pred2idx
        self.label2idx, self.idx2label = self.get_ner_label()

        self.X, self.SEG_ID, self.Y, self.P, self.MASK = self.process_raw_data()

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.SEG_ID[idx], self.Y[idx], self.P[idx], self.MASK[idx]

    def get_ner_label(self):
        label = ['O', 'B-SUB', 'I-SUB', 'B-OBJ', 'I-OBJ', '[PAD]', 'CATEGORY', '[CLS]', '[SEP]']
        label2idx, idx2label = {}, {}
        for i, l in enumerate(label):
            label2idx[l] = i
            idx2label[i] = l
        return label2idx, idx2label

    def process_raw_data(self):
        X, SEG_ID, Y, P, MASK = [], [], [], [], []

        logger.info('Processing NER Dataset....')
        count = 0
        pbar = tqdm(total=len(self.raw_data))
        for line in self.raw_data:
            text = line['Text']
            tokens = self.tokenizer.tokenize(text)
            assert len(line['Predicate']) == len(line['Subject']) == len(line['Object'])

            for p, s, o in zip(line['Predicate'], line['Subject'], line['Object']):
                p_idx = [self.pred2idx[p]] * self.max_len
                tokens = ['[CLS]'] + tokens
                x = [self.tokenizer.convert_tokens_to_ids(t) for t in tokens]
                if len(x) > self.max_len - 1:
                    x = x[0: self.max_len - 1]
                    tokens=tokens[0:self.max_len-1]
                s_begin, s_end = find_sublist(tokens, self.tokenizer.tokenize(s))
                o_begin, o_end = find_sublist(tokens, self.tokenizer.tokenize(o))

                if not (s_begin != None and s_end != None and o_begin != None and o_end != None):
                    count += 1
                    continue


                mask = [1 for i in range(len(x))]

                # initialize y
                y = ['O'] * len(x)
                y[0] = '[CLS]'
                y[s_begin] = 'B-SUB'
                if s_end > s_begin:
                    for i in range(s_begin + 1, s_end + 1):
                        y[i] = 'I-SUB'
                y[o_begin] = 'B-OBJ'
                if o_end > o_begin:
                    for i in range(o_begin + 1, o_end + 1):
                        y[i] = 'I-OBJ'

                seg_id = [0] * (len(x) + 1)
                x = x + [self.tokenizer.convert_tokens_to_ids('[SEP]')]
                mask = mask + [1]
                y = y + ['[SEP]']
                if len(x) <= self.max_len:
                    x += [self.tokenizer.convert_tokens_to_ids('[PAD]')] * (self.max_len - len(x))
                    y += ['[PAD]'] * (self.max_len - len(y))
                    seg_id += [0] * (self.max_len - len(seg_id))
                    mask += [0] * (self.max_len - len(mask))

                # convert tokens to idx
                y = [self.label2idx[t] for t in y]

                assert len(x) == len(seg_id) == len(y) == len(p_idx) == len(mask)
                X.append(x)
                SEG_ID.append(seg_id)
                Y.append(y)
                P.append(p_idx)
                MASK.append(mask)

                pbar.set_description('Not Parse: %.4d' % count)
            pbar.update(1)

        X, SEG_ID, Y, P, MASK = torch.LongTensor(X), torch.LongTensor(SEG_ID), torch.LongTensor(Y), \
                                torch.LongTensor(P), torch.LongTensor(MASK)
        return X, SEG_ID, Y, P, MASK


def get_predicate2idx(filename: str):
    predicates = []
    with jsonlines.open(filename) as reader:
        for line in reader:
            predicates.append(line['predicate'])

    predicates = set(predicates)

    # predicate2idx
    predicate2idx = {predicate: idx for idx, predicate in enumerate(predicates)}
    # idx2predicate
    idx2predicate = {idx: predicate for predicate, idx in predicate2idx.items()}

    return predicate2idx, idx2predicate


def get_char2idx(filename: str):
    chars = []

    with open(filename, 'r') as f:
        for char in f:
            chars.append(char[:-1])

    char2idx = {char: idx for idx, char in enumerate(chars)}
    idx2char = {idx: char for char, idx in char2idx.items()}

    return char2idx, idx2char


def find_sublist(x, subx):
    x_len = len(x)
    subx_len = len(subx)

    for i in range(x_len - subx_len + 1):
        if subx == x[i: i + subx_len]:
            return i, i + subx_len - 1

    return None, None
