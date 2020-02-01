import torch
import torch.nn.functional as F
import numpy as np
import os

from tqdm import tqdm
from utils import logger

class Trainer:
    def __init__(self, train_ner_dataloader, train_p_dataloader, dev_ner_dataloader, \
                 dev_p_dataloader, p_model, ner_model, p_optimizer, ner_optimizer):
        self.train_ner_dataloader = train_ner_dataloader
        self.train_p_dataloader = train_p_dataloader
        self.dev_ner_dataloader = dev_ner_dataloader
        self.dev_p_dataloader = dev_p_dataloader
        self.p_model = p_model
        self.ner_model = ner_model
        self.p_optimizer = p_optimizer
        self.ner_optimizer = ner_optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.p_model.to(self.device)
        self.ner_model.to(self.device)

    def calculate_p_result(self, pred, truth):
        pred = (pred > 0.5).int()

        TP = ((pred == 1) & (truth == 1)).cpu().sum().item()
        TN = ((pred == 0) & (truth == 0)).cpu().sum().item()
        FN = ((pred == 0) & (truth == 1)).cpu().sum().item()
        FP = ((pred == 1) & (truth == 0)).cpu().sum().item()

        p = 0 if TP == FP == 0 else TP / (TP + FP)
        r = 0 if TP == FP == 0 else TP / (TP + FN)
        F1 = 0 if p == r == 0 else 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)

        return (p, r, F1, acc)

    def calculate_ner_result(self, pred, truth):
        # pred = torch.argmax(pred, dim=-1)
        acc = (pred == truth.cpu()).sum().float() / truth.shape[0] / truth.shape[1]
        return acc.item()

    def train_epoch(self, epoch):
        # step1: train p model
        logger.info('Epoch: %2d: Training P Model...' % epoch)
        pbar = tqdm(total = len(self.train_p_dataloader))
        self.p_model.train()

        p_losses, p_p, p_r, p_f1, p_acc = [], [], [], [], []
        for batch in self.train_p_dataloader:
            x, seg_id, y = map(lambda i: i.to(self.device), batch)

            self.p_optimizer.zero_grad()
            out = self.p_model(x, seg_id)
            p_loss = F.binary_cross_entropy(out, y.float())
            p_loss.backward()
            self.p_optimizer.step()

            p_losses.append(p_loss.item())
            p, r, f1, acc = self.calculate_p_result(out, y)
            p_p.append(p)
            p_r.append(r)
            p_f1.append(f1)
            p_acc.append(acc)

            pbar.set_description('Epoch: %2d | LOSS: %2.3f | F1: %1.3f | ACC: %1.3f' % (epoch, p_loss.item(), f1, acc))
            pbar.update(1)
        pbar.close()
        logger.info('Epoch: %2d | LOSS: %2.3f | F1: %1.3f | ACC: %1.3f | PRECISION: %1.3f | RECALL: %1.3f' %
                    (epoch, np.mean(p_losses), np.mean(p_f1), np.mean(p_acc), np.mean(p_p), np.mean(p_r)))

        # step2: train ner model
        logger.info('Epoch %2d: Training NER Model...' % epoch)
        pbar = tqdm(total = len(self.train_ner_dataloader))
        self.ner_model.train()

        ner_losses, ner_acc = [], []
        for batch in self.train_ner_dataloader:
            x, seg_id, y, p, mask = map(lambda i: i.to(self.device), batch)

            self.ner_optimizer.zero_grad()
            out = self.ner_model(x, seg_id, p, mask, self.device)
            ner_loss = self.ner_model.loss_fn(transformer_encode=out, tags=y, output_mask=mask)
            # ner_loss = F.nll_loss(out.reshape(-1, out.shape[2]), y.reshape(-1))
            ner_loss.backward()
            self.ner_optimizer.step()

            ner_losses.append(ner_loss.item())

            predict= self.ner_model.predict(out, mask)
            acc = self.calculate_ner_result(predict, y)
            ner_acc.append(acc)

            pbar.set_description('Epoch: %2d | LOSS: %2.3f | ACC: %1.3f' % (epoch, ner_loss.item(), acc))
            pbar.update(1)
        pbar.close()
        logger.info('Epoch: %2d | LOSS: %2.3f | ACC: %1.3f' % (epoch, np.mean(ner_losses), np.mean(ner_acc)))

        logger.info('FINISH TRAINING EOPCH %2d' % epoch)

    def evaluate_epoch(self, epoch):
        # step1: eval p model
        logger.info('Epoch %2d: Evaluating P Model...' % epoch)
        self.p_model.eval()

        p_losses, p_p, p_r, p_f1, p_acc = [], [], [], [], []
        for batch in self.dev_p_dataloader:
            x, seg_id, y = map(lambda i: i.to(self.device), batch)

            with torch.no_grad:
                out = self.p_model(x, seg_id)
            p_loss = F.binary_cross_entropy(out, y.float())

            p_losses.append(p_loss.item())
            p, r, f1, acc = self.calculate_p_result(out, y)
            p_p.append(p)
            p_r.append(r)
            p_f1.append(f1)
            p_acc.append(acc)

        logger.info('Epoch: %2d | LOSS: %2.3f | F1: %1.3f | ACC: %1.3f | PRECISION: %1.3f | RECALL: %1.3f' %
                    (epoch, np.mean(p_losses), np.mean(p_f1), np.mean(p_acc), np.mean(p_p), np.mean(p_r)))

        # step2: train ner model
        logger.info('Epoch %2d: Evaluating NER Model...' % epoch)
        self.ner_model.eval()

        ner_losses, ner_acc = [], []
        for batch in self.dev_ner_dataloader:
            x, seg_id, y = map(lambda i: i.to(self.device), batch)

            with torch.no_grad:
                out = self.ner_model(x, seg_id)
            ner_loss = F.nll_loss(out.reshape(-1, out.shape[2]), y.reshape(-1))

            ner_losses.append(ner_loss.item())
            acc = self.calculate_ner_result(out, y)
            ner_acc.append(acc)

        logger.info('Epoch: %2d | LOSS: %2.3f | ACC: %1.3f' % (epoch, np.mean(ner_losses), np.mean(ner_acc)))

        logger.info('FINISH EVALUATING EOPCH %2d' % epoch)

    def train(self, num_epoch, save_path):
        for epoch in range(num_epoch):
            self.train_epoch(epoch)
            self.evaluate_epoch(epoch)

            # save state dict
            path = os.path.join(save_path, 'state_%2d_epoch.pt' % epoch)
            self.save_dict(path)

    def save_dict(self, save_path):
        state_dict = {
            'p_model': self.p_model.state_dict(),
            'ner_model': self.ner_model.state_dict(),
            'p_optimizer': self.p_optimizer.state_dict(),
            'ner_optimizer': self.ner_optimizer.state_dict()
        }

        torch.save(state_dict, save_path)

    def load_dict(self, path):
        state_dict = torch.load(path)

        self.p_model.load_state_dict(state_dict['p_model'])
        self.ner_model.load_state_dict(state_dict['ner_model'])
        self.p_optimizer.load_state_dict(state_dict['p_optimizer'])
        self.ner_optimizer.load_state_dict(state_dict['ner_optimizer'])
