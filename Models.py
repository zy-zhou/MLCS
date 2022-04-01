# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:35:43 2019

@author: Zhou
"""

from DecoderWrappers import SampleDecodingWrapper, BeamSearchWrapper,\
                            EnsembleSamplDecWrapper, EnsembleBmSrchWrapper
from Utils import save, sequence_loss, batch_bleu, batch_meteor, batch_rouge,\
                  perplexity, tuple_map
from Data import id2word, pad
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from learn2learn.algorithms import MAML

beam_width = 5
n_best = 1
sampling_temp = 1.
sampling_topk = -1
max_iter = 30
length_penalty = 1.
coverage_penalty = 0.

class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.is_ensemble = False
    
    def forward(self, src_batch, src_lengths=None, tgt_batch=None, weights=None):
        tgt_batch = tgt_batch[:,:-1]
        memory, final_state = self.encoder(src_batch, src_lengths)
        logits, attn_history = self.decoder(tgt_batch, final_state, memory, src_lengths, weights)
        return logits, attn_history

class Translator(object):
    def __init__(self, model, sampling_temp=sampling_temp, sampling_topk=sampling_topk,
                 beam_width=beam_width, n_best=n_best, max_iter=max_iter,
                 length_penalty=length_penalty, coverage_penalty=coverage_penalty,
                 metrics=['loss', 'bleu'], unk_replace=False, smooth=3):
        self.model = model
        self.metrics = metrics
        self.unk_replace = unk_replace
        self.smooth = smooth
        
        if not beam_width or beam_width == 1:
            if model.is_ensemble:
                self.wrapped_decoder = EnsembleSamplDecWrapper(
                        model.decoder, sampling_topk, max_iter)
            else:
                self.wrapped_decoder = SampleDecodingWrapper(
                        model.decoder, sampling_temp, sampling_topk, max_iter)
        else:
            if model.is_ensemble:
                self.wrapped_decoder = EnsembleBmSrchWrapper(
                        model.decoder, beam_width, n_best, max_iter, length_penalty, coverage_penalty)
            else:
                self.wrapped_decoder = BeamSearchWrapper(
                        model.decoder, beam_width, n_best, max_iter, length_penalty, coverage_penalty)

    @property
    def metrics(self):
        return self._metrics
    
    @metrics.setter
    def metrics(self, metrics):
        metrics = set(metrics)
        all_metrics = {'loss', 'bleu', 'rouge', 'meteor'}
        if not metrics.issubset(all_metrics):
            raise ValueError('Unkown metric(s): ' + str(metrics.difference(all_metrics)))
        self._metrics = metrics
    
    def val_loss(self, final_state, memory, src_lengths, tgt_batch, weights):
        outputs, _ = self.model.decoder(tgt_batch[:,:-1], final_state, memory, src_lengths)
        loss = sequence_loss(outputs, tgt_batch[:,1:], is_probs=self.model.is_ensemble,
                             pad_id=self.wrapped_decoder.pad_id)
        return loss.item()
    
    def translate_batch(self, src_batch, src_lengths=None, tgt_batch=None, *args):
        weights = args[0] if self.weights else None
        raw_batches = args[-1] if self.raw_data else [None]
        reports = dict(scores=None, attn_history=None)
        with torch.no_grad():
            memory, final_state = self.model.encoder(src_batch, src_lengths)
            if 'loss' in self._metrics:
                reports['loss'] = self.val_loss(final_state, memory, src_lengths, tgt_batch, weights)
            predicts, reports['scores'], reports['attn_history'] = \
            self.wrapped_decoder(final_state, memory, src_lengths, weights)
        
        if type(self.wrapped_decoder) in {BeamSearchWrapper, EnsembleBmSrchWrapper}:
            predicts = [b[0] for b in predicts]
            reports['scores'] = [b[0] for b in reports['scores']]
            if reports['attn_history'][0]:
                reports['attn_history'] = [b[0] for b in reports['attn_history']]
        
        predicts = id2word(predicts, self.model.decoder.field,
                           (raw_batches[0], reports['attn_history']), 
                           replace_unk=self.unk_replace)
        if 'bleu' in self._metrics:
            reports['bleu'] = batch_bleu(predicts, raw_batches[-1], self.smooth) * 100
        predicts = [' '.join(s) for s in predicts]
        
        if not self._metrics.isdisjoint({'rouge', 'meteor'}):
            targets = [' '.join(s) for s in raw_batches[-1]]
            if 'rouge' in self._metrics:
                rouge = batch_rouge(predicts, targets)
                reports['rouge'] = rouge['rouge-l']['f'] * 100
            if 'meteor' in self._metrics:
                reports['meteor'] = batch_meteor(predicts, targets) * 100
        
        return predicts, reports
    
    def init_generator(self, data_gen):
        data_gen.raw_data = self.unk_replace or self._metrics.difference({'loss'})
        self.raw_data = data_gen.raw_data
        self.weights = self.model.is_ensemble and data_gen.weights
    
    def __call__(self, batches, save_path=None):
        self.init_generator(batches)
        self.model.eval()
        results = []
        reports = defaultdict(float, scores=[], attn_history=[])
        
        pbar = tqdm(batches, desc='Translating...')
        for batch in pbar:
            predicts, reports_ = self.translate_batch(*batch)
            pbar.set_postfix({metric: reports_[metric] for metric in self._metrics})
            results.extend(predicts)
            for metric in self._metrics:
                reports[metric] += reports_[metric]
#            reports['scores'].extend(reports_['scores'])
            # reports['attn_history'].extend(reports_['attn_history'])
        
        for metric in self._metrics:
            reports[metric] /= len(batches)
            print('total {}: {:.2f}'.format(metric, reports[metric]))
            if metric == 'loss':
                reports['ppl'] = perplexity(reports[metric])
                print('total ppl: {:.2f}'.format(reports['ppl']))
        if save_path is not None:
            save(results, save_path)
        return results, reports

class MetaTranslator(Translator):
    def __init__(self, model, adapt_steps=1, temperature=0, **kwargs):
        self.device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
        self.adapt_steps = adapt_steps
        self.temperature = temperature
        self.pad_id = model.decoder.field.vocab.stoi[pad]
        super(MetaTranslator, self).__init__(model, **kwargs)
    
    def fast_adapt(self, adapt_batch, weights):
        learner = self.model.clone()
        learner.train()
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
        learner.eval()
        self.model = learner
        self.wrapped_decoder.decoder = learner.decoder
    
    def translate_batch(self, adapt_batches, eval_batches, sims, raw_batches=None):
        predicts = []
        reports = defaultdict(float, scores=[], attn_history=[])
        for idx, batch in enumerate(zip(adapt_batches, eval_batches, sims)):
            adapt_batch, eval_batch, weights = tuple_map(
                lambda x: x.to(self.device) if type(x) is torch.Tensor else x, batch)
            model = self.model
            self.fast_adapt(adapt_batch, weights)
            if self.raw_data:
                eval_batch = eval_batch + (raw_batches[idx],)
            predict, report = Translator.translate_batch(self, *eval_batch)
            predicts.extend(predict)
            for metric in self._metrics:
                reports[metric] += report[metric]
            self.model = model
        
        for metric in self._metrics:
            reports[metric] /= len(adapt_batches)
        return predicts, reports
    
    def init_generator(self, data_gen):
        super(MetaTranslator, self).init_generator(data_gen)
        data_gen.device = 'cpu'