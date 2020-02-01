import argparse

parser = argparse.ArgumentParser()

# global setting
parser.add_argument('--train_file', type=str, default='data/train_data.json')
parser.add_argument('--dev_file', type=str, default='data/dev_data.json')
parser.add_argument('--test_file', type=str, default='data/test_data.json')
parser.add_argument('--test_result_file', type=str, default='save/result.csv')
parser.add_argument('--train_ner_data', type=str, default='data/train_ner.pt')
parser.add_argument('--dev_ner_data', type=str, default='data/dev_ner.pt')
parser.add_argument('--train_p_data', type=str, default='data/train_p.pt')
parser.add_argument('--dev_p_data', type=str, default='data/dev_p.pt')
parser.add_argument('--schemas', type=str, default='data/all_50_schemas')
parser.add_argument('--predicate2idx', type=str, default='data/predicate2idx.pt')
parser.add_argument('--idx2predicate', type=str, default='data/idx2predicate.pt')
parser.add_argument('--idx2label', type=str, default='data/idx2label.pt')
parser.add_argument('--do_preprocess', action='store_true', help='whether to do preprocess', default=False)
parser.add_argument('--do_train', action='store_true', help='whether to do train', default=False)
parser.add_argument('--do_predict', action='store_true', help='whether to do predict', default=False)
parser.add_argument('--bert_path', type=str, default='./pretrained_bert_model/')

# about preprocess
parser.add_argument('--max_len', type=int, default=200)

# about model
parser.add_argument('--bert_dim', type=int, default=768)
parser.add_argument('--drop_p', type=float, default=0.3)

# about train
parser.add_argument('--num_epoch', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--p_lr', type=float, default=0.0001)
parser.add_argument('--ner_lr', type=float, default=0.0001)

# about predict
parser.add_argument('--state_dict', type=str, default='save/state_0_epoch.pt')

# about log and saving models
parser.add_argument('--log_file', type=str, default='./log/out.log')
parser.add_argument('--save_path', type=str, default='./save')


class Hyperparams:
    '''Hyperparameters'''
    hidden_units = 768*2  # alias = C
    num_blocks = 3  # number of encoder/decoder blocks
    num_heads = 4
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.