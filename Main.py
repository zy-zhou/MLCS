# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:47:52 2019

@author: Zhou
"""
import torch
from Utils import load
from Data import load_data
from Modules import BasicDecoder, RNNEncoder
from Models import Model, MetaTranslator
from Train import MetaTrainer
import warnings
warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

meta_epoches = 15
meta_batch_size = 20
adapt_lr = 0.4
n = 5

if __name__ == '__main__':
    from Utils import batch_bleu, batch_meteor, batch_rouge
    
    fields, train_gen = load_data('train', 'meta', batch_size=meta_batch_size, k=1, epsilon=0.7,
                                  device=device, meta_weights=True)
    _, val_gen = load_data('valid', 'meta', batch_size=meta_batch_size, k=10, epsilon=0.7,
                           device=device, meta_weights=True)

    e = RNNEncoder(fields[0], bidirectional=True)
    d = BasicDecoder(fields[1], memory_dim=e.units * 2, glob_attn='mul')
    model = Model(e, d)
    model = model.to(device)
    checkpoint = torch.load('checkpoints/nmt.pt')
    model.load_state_dict(checkpoint['model'])
    
    trainer = MetaTrainer(model, epoches=meta_epoches, temperature=0,
                          metrics=['bleu'], smooth=0, patience=4, save_per_epoch=False,
                          beam_width=5, length_penalty=1, val_metric='bleu',
                          adapt_lr=adapt_lr, first_order=True,
                          save_path='checkpoints/nmt_meta.pt')
    reports = trainer(train_gen, val_gen)

####################################################################################
    _, test_gen = load_data('test', 'meta', batch_size=meta_batch_size, k=10, epsilon=0.7, meta_weights=True)
    
    trainer = MetaTrainer(model, adapt_lr=adapt_lr / (n - 1) if n > 2 else 2/3 * adapt_lr,
                          load_path='checkpoints/nmt_meta.pt')
    evaluator = MetaTranslator(trainer.model, metrics=[], adapt_steps=n, unk_replace=False)
    predicts, reports = evaluator(test_gen, save_path='predicts/nmt_meta.txt')
    
####################################################################################
    hyp = [s.split() for s in predicts]
    ref = load('data/preprocessed/test.nl.json', is_json=True)
    bleu_4 = batch_bleu(hyp, ref, smooth_method=0)
    print('BLEU-4: {:.2f}'.format(bleu_4 * 100))
    bleu_s = batch_bleu(hyp, ref, smooth_method=3)
    print('Smoothed BLEU-4: {:.2f}'.format(bleu_s * 100))
    hyp = predicts
    ref = [' '.join(s) for s in ref]
    rouge = batch_rouge(hyp, ref)
    print('ROUGE-L: {:.2f}'.format(rouge['rouge-l']['f'] * 100))
    meteor = batch_meteor(hyp, ref)
    print('METEOR: {:.2f}'.format(meteor * 100))
    