import os
import numpy as np

from torch.utils.data import DataLoader
from utils import *
from args import parser
from model import P_Model, NER_Model
from trainer import Trainer
from predicate import Predictor

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_preprocess():
    # get predicate2idx and char2idx
    predicate2idx, idx2predicate = get_predicate2idx(args.schemas)

    torch.save(predicate2idx, args.predicate2idx)
    torch.save(idx2predicate, args.idx2predicate)

    # get the dataset
    train_processor, dev_processor = KG_DataProcessor(args.train_file, 173108, predicate2idx), \
            KG_DataProcessor(args.dev_file, 21639, predicate2idx)
    train_data, dev_data = train_processor.get_data(), dev_processor.get_data()
    train_ner_dataset, dev_ner_dataset = NERDataset(train_data, args.max_len, args.bert_path, predicate2idx), \
            NERDataset(dev_data, args.max_len, args.bert_path, predicate2idx)
    train_p_dataset, dev_p_dataset = PDataset(train_data, args.max_len, args.bert_path, predicate2idx), \
            PDataset(dev_data, args.max_len, args.bert_path, predicate2idx)

    torch.save(train_ner_dataset, args.train_ner_data)
    torch.save(dev_ner_dataset, args.dev_ner_data)
    torch.save(train_p_dataset, args.train_p_data)
    torch.save(dev_p_dataset, args.dev_p_data)

def train():
    # prepare data
    logger.info('PREPARING DATA...')
    predicate2idx, idx2predicate = get_predicate2idx(args.schemas)

    train_p_dataset = torch.load(args.train_p_data)
    train_ner_dataset = torch.load(args.train_ner_data)
    dev_p_dataset = torch.load(args.dev_p_data)
    dev_ner_dataset = torch.load(args.dev_ner_data)

    train_p_dataloader = DataLoader(train_p_dataset, batch_size=args.batch_size, shuffle=True)
    train_ner_dataloader = DataLoader(train_ner_dataset, batch_size=args.batch_size, shuffle=True)
    dev_p_dataloader = DataLoader(dev_p_dataset, batch_size=args.batch_size, shuffle=True)
    dev_ner_dataloader = DataLoader(dev_ner_dataset, batch_size=args.batch_size, shuffle=True)

    logger.info('FINISH PREPARING DATA!')

    # prepare model
    logger.info('PREPARING MODEL & OPTIMIER...')

    p_model = P_Model(args.bert_path, args.bert_dim, len(train_p_dataset.pred2idx), args.drop_p, args.max_len).to(device)
    ner_model = NER_Model(args.bert_path, args.bert_dim, len(train_ner_dataset.label2idx), args.drop_p, len(predicate2idx)).to(device)

    p_optimizer = torch.optim.Adam(params=p_model.parameters(), lr=args.p_lr)
    ner_optimizer = torch.optim.Adam(params=ner_model.parameters(), lr=args.ner_lr)

    logger.info('FINISH PREPARING MODEL & OPTIMIZER!')

    # training and evaluating
    logger.info('TRAINING...')

    trainer =  Trainer(train_ner_dataloader, train_p_dataloader, dev_ner_dataloader, dev_p_dataloader, \
                       p_model, ner_model, p_optimizer, ner_optimizer)
    trainer.train(args.num_epoch, args.save_path)

def predict():
    # prepare model and other essential variables
    if not os.path.exists(args.state_dict):
        raise Exception('Please set arguments with correct state_dict')

    # get idx2pred
    idx2pred = torch.load(args.idx2predicate)
    idx2ner = torch.load(args.idx2label)

    p_model, ner_model = load_dict(args.state_dict, len(idx2pred), len(idx2ner))

    # predict
    predictor = Predictor(args.test_file, args.test_result_file, p_model, ner_model, idx2pred, \
                          idx2ner, args.bert_path, args.max_len, args.batch_size)
    predictor.predict()


def main():
    if args.do_preprocess:
        data_preprocess()

    if args.do_train:
        train()

    if args.do_predict:
        predict()

if __name__ == '__main__':
    main()