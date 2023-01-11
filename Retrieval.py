# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 20:00:45 2021

@author: Zhou
"""

import torch
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import FastText, Word2Vec, TfidfModel, LsiModel
from gensim.similarities import MatrixSimilarity, SparseMatrixSimilarity
from Levenshtein import distance
from Data import DataGenerator
from Utils import parallel
from tqdm import tqdm

def _leven_sim(query, corpus):
    ''' query: a single string
        corpus: list of strings to compare'''
    sims = [1 - distance(query, doc) / max(len(query), len(doc)) for doc in corpus]
    return sims

def leven_sim(queries, corpus, workers=1):
    desc = 'Computing similarities...'
    data = list(zip(queries, corpus))
    if workers > 1 and len(queries) > 2000:
        sims = parallel(_leven_sim, data, workers=workers, star=True)
    else:
        sims = []
        for query, corpus_ in tqdm(data, desc=desc):
            sims.append(_leven_sim(query, corpus_))
    return sims

def sims_topk(sims, k=10, include_self=False):
    if not include_self:
        topk, topk_idx = sims.topk(k, 1)
        return topk, topk_idx
    else:
        topk, topk_idx = sims.topk(k + 1, 1)
        return topk[:,1:], topk_idx[:,1:]

def cosine_nns(query_vecs, corpus_vecs, k, include_self, batch_size=16):
    assert type(corpus_vecs) is torch.Tensor
    size = query_vecs.shape[0], corpus_vecs.shape[0], corpus_vecs.shape[1]
    query_vecs = query_vecs.unsqueeze(1).expand(*size)
    corpus_vecs = corpus_vecs.unsqueeze(0).expand(*size)
    # batch_size * corpus_size * n_dim
    query_batches = DataGenerator.batch(query_vecs, batch_size)
    corpus_batches = DataGenerator.batch(corpus_vecs, batch_size)
    topk, topk_idx = [], []
    for query_batch, corpus_batch in tqdm(zip(query_batches, corpus_batches),
                                          total=int(np.ceil(size[0] / batch_size)),
                                          desc='Computing similarities...'):
        sims = torch.cosine_similarity(query_batch, corpus_batch, -1)
        values, indices = sims_topk(sims, k=k, include_self=include_self)
        topk.append(values.to('cpu'))
        topk_idx.append(indices.to('cpu'))
    topk = torch.cat(topk)
    topk_idx = torch.cat(topk_idx)
    return topk, topk_idx

class Doc2Vec(object):
    def __init__(self, corpus, embed_dim=128, model='fasttext', **kwargs):
        if model == 'fasttext':
            self.model = FastText(sentences=corpus, vector_size=embed_dim, **kwargs)
        else:
            self.model = Word2Vec(sentences=corpus, vector_size=embed_dim, **kwargs)
    
    def get_vecs(self, docs, pooling='avg'):
        doc_vecs = []
        for doc in docs:
            vecs = [self.model.wv[w] for w in doc]
            if pooling == 'avg':
                vec = np.stack(vecs).mean(0)
            else:
                vec = np.stack(vecs).max(0)
            doc_vecs.append(vec)
        return torch.from_numpy(np.stack(doc_vecs))
    
class SparseDoc2Vec(object):
    def __init__(self, corpus, model='tfidf', lsi=False, num_topics=500, **kwargs):
        self.dictionary = Dictionary(corpus)
        self.mode = model
        corpus = [self.dictionary.doc2bow(d) for d in corpus]
        if model == 'tfidf':
            self.tfidf_model = TfidfModel(corpus, **kwargs)
            corpus = self.tfidf_model[corpus]
        if lsi:
            self.lsi_model = LsiModel(corpus, id2word=self.dictionary,
                                      num_topics=num_topics, **kwargs)
    
    def get_vecs(self, docs, lsi=False):
        docs = [self.dictionary.doc2bow(d) for d in docs]
        if self.mode == 'tfidf':
            docs = self.tfidf_model[docs]
        if lsi:
            assert hasattr(self, 'lsi_model')
            docs = self.lsi_model[docs]
        return docs

    def sparse_cosine(self, queries, corpus):
        ''' Expecting tokenized raw inputs '''
        corpus = self.get_vecs(corpus, lsi=False)
        queries = self.get_vecs(queries, lsi=False)
        index = SparseMatrixSimilarity(corpus, num_features=len(self.dictionary))
        return torch.from_numpy(index[queries])

class RNNDoc2Vec(object):
    def __init__(self, model, batch_size=100):
        self.model = model
        self.batch_size = batch_size
        self.device = 'cuda' if next(self.model.parameters()).is_cuda else 'cpu'
    
    def process_batch(self, docs):
        lengths = list(map(len, docs))
        lengths = torch.tensor(lengths, device=self.device)
        batch = self.model.field.process(docs, device=self.device)
        return batch, lengths
    
    def get_vecs(self, docs, pooling='max'):
        results = []
        batches = DataGenerator.batch(docs, self.batch_size)
        self.model.eval()
        with torch.no_grad():
            for raw_batch in tqdm(batches, total=int(np.ceil(len(docs) / self.batch_size)),
                                  desc='Caching model outputs...'):
                batch, lengths = self.process_batch(raw_batch)
                vecs, _ = self.model(batch, lengths)
                vecs = vecs.max(1).values if pooling == 'max' else vecs.mean(1)
                results.append(vecs.to('cpu'))
            results = torch.cat(results)
        return results

if __name__ == '__main__':
    from Utils import load
    from Modules import RNNEncoder, BasicDecoder
    from Models import Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    code_field = torch.load('data/preprocessed/code_field.pkl')
    nl_field = torch.load('data/preprocessed/nl_field.pkl')
    e = RNNEncoder(code_field, bidirectional=True)
    d = BasicDecoder(nl_field, memory_dim=e.units * 2, glob_attn='mul')
    model = Model(e, d)
    model = model.to('cuda')
    checkpoint = torch.load('checkpoints/nmt.pt')
    model.load_state_dict(checkpoint['model'])
    doc2vec = RNNDoc2Vec(model.encoder)
    
    prefix = ('train', 'valid', 'test')
    for pre in prefix:
        codes = load('data/preprocessed/' + pre + '.code.json', is_json=True)
        rnn_vecs = doc2vec.get_vecs(codes)
        torch.save(rnn_vecs, 'data/retrieved/' + pre + '.rnn.vecs.pt')
    
    corpus_vecs = torch.load('data/retrieved/train.rnn.vecs.pt').to(device)
    for pre in prefix:
        if pre == 'train':
            query_vecs = corpus_vecs
        else:
            query_vecs = torch.load('data/retrieved/' + pre + '.rnn.vecs.pt').to(device)
    
        topk, topk_idx = cosine_nns(query_vecs, corpus_vecs, k=10, include_self='train' in pre)
        torch.save(dict(topk=topk, idx=topk_idx), 'data/retrieved/'+ pre + '.rnn.topk.pt')
    
    corpus = load('data/preprocessed/train.code.json', is_json=True)
    corpus = [' '.join(s) for s in corpus]
    for pre in prefix:
        if pre == 'train':
            queries = corpus
        else:
            queries = load('data/preprocessed/' + pre + '.code.json', is_json=True)
            queries = [' '.join(s) for s in queries]
        d = torch.load('data/retrieved/' + pre + '.rnn.topk.pt')
        corpus_ = [[corpus[i] for i in k] for k in d['idx']]
        sims = leven_sim(queries, corpus_, workers=4)
        sims = np.array(sims)
        print(sims.shape)
        d['sims'] = sims
        torch.save(d, 'data/retrieved/' + pre + '.rnn.topk.pt')
