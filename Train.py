# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:35:27 2019

@author: Zhou
"""
from Utils import sequence_loss, perplexity, tuple_map
from Models import Translator, MetaTranslator
from Data import pad
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from learn2learn.algorithms import MAML
from abc import abstractmethod

epoches = 70
lr = 0.001
optimizer = 'adam'
max_grad_norm = 5
lr_decay = None

def get_optimizer(optimizer, lr, params):
    params = filter(lambda p: p.requires_grad, params)
    if optimizer == 'sgd':
        return optim.SGD(params, lr=lr)
    elif optimizer == 'nag':
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    elif optimizer == 'adagrad':
        return optim.Adagrad(params, lr=lr)
    elif optimizer == 'adadelta':
        return optim.Adadelta(params, lr=lr)
    elif optimizer == 'adam':
        return optim.Adam(params, lr=lr)
    else:
        raise ValueError('Invalid optimizer type: ' + optimizer)

class Trainer(object):
    def __init__(self, model, epoches=epoches, optimizer=optimizer, lr=lr,
                 max_grad_norm=max_grad_norm, lr_decay=lr_decay, metrics=['loss'],
                 val_metric='loss', save_path=None, load_path=None, patience=3,
                 save_per_epoch=True, **kwargs):
        self.model = model
        self.pad_id = model.decoder.field.vocab.stoi[pad]
        self.epoches = epoches
        self.save_path = save_path
        self.save_per_epoch = save_per_epoch
        self.patience = patience if save_path else float('inf')
        self.optimizer = get_optimizer(optimizer, lr, model.parameters())
        if lr_decay:
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, lr_decay)
        
        self.metrics = set(metrics)
        assert val_metric in metrics
        self.val_metric = val_metric
        if self.metrics.difference({'loss'}):
            kwargs['metrics'] = self.metrics
            self.evaluator = Translator(model, **kwargs)
        
        self.device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
        if load_path:
            self.load_states(load_path)
        else:
            self.max_grad_norm = max_grad_norm
            self.curr_epoch = self.curr_iter = self.best_epoch = 0
            self.best_score = float('-inf')
            self.log = defaultdict(list)
        
    def save_states(self):
        print('Saving model and settings...')
        checkpoint = dict(model=self.model.state_dict(),
                          optimizer=self.optimizer.state_dict(),
                          max_grad_norm = self.max_grad_norm,
                          epoch=self.curr_epoch,
                          iteration=self.curr_iter,
                          log=self.log,
                          best_epoch=self.best_epoch,
                          best_score=self.best_score)
        if hasattr(self, 'scheduler'):
            checkpoint['scheduler'] = self.scheduler.state_dict()
        
        new_path = self.save_path
        if self.save_per_epoch:
            new_path = new_path.split('.')
            new_path[0] = new_path[0] + '_epoch' + str(self.curr_epoch)
            new_path = '.'.join(new_path)
        torch.save(checkpoint, new_path)
    
    def load_states(self, load_path):
        print('Loading model and settings from checkpoint...')
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.max_grad_norm = checkpoint['max_grad_norm']
        self.curr_epoch = checkpoint['epoch']
        self.curr_iter = checkpoint['iteration']
        self.log = checkpoint['log']
        self.best_epoch = checkpoint['best_epoch']
        self.best_score = checkpoint['best_score']
        if 'scheduler' in checkpoint.keys() and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler'])
    
    def validate_epoch(self, val_iter):
        if self.metrics.difference({'loss'}):
            _, reports = self.evaluator(val_iter)
            del reports['scores'], reports['attn_history']
            return reports
            
        self.model.eval()
        reports = dict(loss=0)
        pbar = tqdm(val_iter, desc='Validating epoch ' + str(self.curr_epoch))
        with torch.no_grad():
            for batch in pbar:
                outputs, _ = self.model(*batch)
                loss = sequence_loss(outputs, batch[-1][:,1:], is_probs=self.model.is_ensemble,
                                     pad_id=self.pad_id)
                reports['loss'] += loss.item()
                pbar.set_postfix(loss=loss.item())
        
        reports['loss'] /= len(val_iter)
        print('val loss: {:.2f}'.format(reports['loss']))
        reports['ppl'] = perplexity(reports['loss'])
        print('val ppl: {:.2f}'.format(reports['ppl']))
        torch.cuda.empty_cache()
        return reports
    
    @abstractmethod
    def train_epoch(self, train_iter):
        pass
    
    @abstractmethod
    def init_generator(self, train_iter, val_iter=None):
        pass
    
    def __call__(self, train_iter, val_iter=None):
        self.init_generator(train_iter, val_iter)
        
        for epoch in range(self.epoches):
            reports = self.train_epoch(train_iter)
            for key, value in reports.items():
                self.log[key].append(value)
            self.curr_epoch += 1
            
            if val_iter is not None:
                reports = self.validate_epoch(val_iter)
                for key, value in reports.items():
                    self.log['val_' + key].append(value)
                
                score = - reports['ppl'] if self.val_metric == 'loss' else reports[self.val_metric]
                if score > self.best_score:
                    self.best_epoch = self.curr_epoch
                    self.best_score = score
                    if self.save_path:
                        self.save_states()
                elif self.curr_epoch - self.best_epoch >= self.patience:
                    print('Early stopped at epoch ' + str(self.curr_epoch))
                    print('Best validating score reached at epoch ' + str(self.best_epoch))
                    break
            elif self.save_path:
                self.save_states()
            
        return self.log
    
class TeacherForcing(Trainer):
    def train_step(self, src_batch, src_lengths, tgt_batch):
        self.optimizer.zero_grad()
        outputs, _ = self.model(src_batch, src_lengths, tgt_batch)
        loss = sequence_loss(outputs, tgt_batch[:,1:], is_probs=self.model.is_ensemble,
                             pad_id=self.pad_id)
        
        loss.backward()
        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()
    
    def train_epoch(self, train_iter):
        self.model.train()
        reports = dict(loss=0)
        
        pbar = tqdm(train_iter, desc='Training epoch ' + str(self.curr_epoch + 1))
        for batch in pbar:
            loss = self.train_step(*batch)
            self.curr_iter += 1
            reports['loss'] += loss
            pbar.set_postfix(loss=loss)
#            torch.cuda.empty_cache()
        
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
        reports['loss'] /= len(train_iter)
        print('train loss: {:.2f}'.format(reports['loss']))
        reports['ppl'] = perplexity(reports['loss'])
        print('train ppl: {:.2f}'.format(reports['ppl']))
        reports['lr'] = self.optimizer.param_groups[0]['lr']
        print('learning rate: {:.5f}'.format(reports['lr']))
        return reports
    
    def init_generator(self, train_iter, val_iter=None):
        train_iter.raw_data = False
        if val_iter is not None:
            assert len(self.metrics) > 0
            val_iter.raw_data = True if self.metrics.difference({'loss'}) else False
    
class MetaTrainer(TeacherForcing):
    def __init__(self, model, epoches=epoches, optimizer=optimizer, lr=lr,
                 adapt_lr=0.4, first_order=True, adapt_steps=1, temperature=0,
                 max_grad_norm=max_grad_norm, lr_decay=lr_decay, metrics=['loss'],
                 val_metric='loss', save_path=None, load_path=None, patience=3,
                 save_per_epoch=True, **kwargs):
        model = MAML(model, adapt_lr, first_order)
        self.adapt_steps = adapt_steps
        self.temperature = temperature
        super(MetaTrainer, self).__init__(model, epoches, optimizer, lr, max_grad_norm,
                                          lr_decay, metrics, val_metric, save_path,
                                          load_path, patience, save_per_epoch)
        if self.metrics.difference({'loss'}):
            kwargs['metrics'] = self.metrics
            self.evaluator = MetaTranslator(model, adapt_steps, temperature, **kwargs)
    
    def fast_adapt(self, adapt_batch, eval_batch, weights):
        adapt_batch, eval_batch, weights = tuple_map(
            lambda x: x.to(self.device) if type(x) is torch.Tensor else x,
            (adapt_batch, eval_batch, weights))
        learner = self.model.clone()
        for step in range(self.adapt_steps):
            outputs, _ = learner(*adapt_batch)
            adapt_loss = sequence_loss(outputs, adapt_batch[-1][:,1:],
                                       is_probs=learner.is_ensemble,
                                       pad_id=self.pad_id, reduction='none')
            if self.temperature == 0:
                tau = 1. / weights.shape[0]
            else:
                tau = self.temperature
            adapt_loss = torch.sum(F.softmax(weights / tau, 0) * (weights + 0.5) * adapt_loss)
            learner.adapt(adapt_loss)
        
        outputs, _ = learner(*eval_batch)
        eval_loss = sequence_loss(outputs, eval_batch[-1][:,1:],
                                  is_probs=learner.is_ensemble,
                                  pad_id=self.pad_id)
        return eval_loss
            
    def validate_epoch(self, val_iter):
        if self.metrics.difference({'loss'}):
            _, reports = self.evaluator(val_iter)
            del reports['scores'], reports['attn_history']
            return reports
        
        reports = dict(loss=0)
        pbar = tqdm(val_iter, desc='Validating epoch ' + str(self.curr_epoch))
        for adapt_batches, eval_batches, sims in pbar:
            loss = 0
            for adapt_batch, eval_batch, weights in zip(adapt_batches, eval_batches, sims):
                eval_loss = self.fast_adapt(adapt_batch, eval_batch, weights)
                loss += eval_loss.item()
            loss /= len(adapt_batches)
            reports['loss'] += loss
            pbar.set_postfix(loss=loss)
        
        reports['loss'] /= len(val_iter)
        print('val loss: {:.2f}'.format(reports['loss']))
        reports['ppl'] = perplexity(reports['loss'])
        print('val ppl: {:.2f}'.format(reports['ppl']))
        torch.cuda.empty_cache()
        return reports
    
    def train_step(self, adapt_batches, eval_batches, sims):
        loss = 0
        self.optimizer.zero_grad()
        for adapt_batch, eval_batch, weights in zip(adapt_batches, eval_batches, sims):
            eval_loss = self.fast_adapt(adapt_batch, eval_batch, weights)
            eval_loss.backward()
            loss += eval_loss.item()
        
        for p in self.model.parameters():
            p.grad.data.mul_(1. / len(adapt_batches))
        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss / len(adapt_batches)
    
    def init_generator(self, train_iter, val_iter=None):
        super(MetaTrainer, self).init_generator(train_iter, val_iter)
        train_iter.device = 'cpu'
        if val_iter is not None:
            val_iter.device = 'cpu'

if __name__ == '__main__':
    from Data import load_data
    from Modules import RNNEncoder, BasicDecoder
    from Models import Model
    import warnings
    warnings.filterwarnings("ignore")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    fields, train_gen = load_data('train', device=device)
    _, val_gen = load_data('valid', device=device)
    
    e = RNNEncoder(fields[0], bidirectional=True)
    d = BasicDecoder(fields[1], memory_dim=e.units * 2, glob_attn='mul')
    model = Model(e, d)
    model = model.to(device)
    
    trainer = TeacherForcing(model, epoches=epoches, optimizer='adam',
                              metrics=['rouge', 'bleu'], smooth=0, patience=4, save_per_epoch=False,
                              beam_width=5, length_penalty=1, val_metric='bleu',
                              save_path='checkpoints/nmt.pt')
    reports = trainer(train_gen, val_gen)
