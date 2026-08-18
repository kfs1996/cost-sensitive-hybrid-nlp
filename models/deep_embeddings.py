import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
import os
import re
import ssl
import urllib3
import requests

# Fix Windows SSL Verification Issues for HuggingFace/Stanford downloads
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_orig_send = requests.Session.send
def _patched_send(self, request, **kwargs):
    kwargs['verify'] = False
    return _orig_send(self, request, **kwargs)
requests.Session.send = _patched_send

# Cache models in global memory
_GLOVE = None
_BERT_MODEL = None
_BERT_TOK = None
_SBERT_MODEL = None
_MPNET_MODEL = None

# Global Dictionary to cache identical texts so we only extract them ONCE across 500 algorithms
_TEXT_CACHE = {'GloVe': {}, 'BERT': {}, 'SBERT': {}, 'MPNet': {}}

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # convert class string labels to integers
        self.labels = np.unique(y)
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}
        
        y_idx = [self.label_to_idx[label] for label in y]
        self.y = torch.tensor(y_idx, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def simple_tokenize(text):
    return re.findall(r'\b\w+\b', str(text).lower())

def get_deep_embeddings(texts_train, texts_test, embed_type):
    """
    Extract REAL sequence embeddings for the baseline paper.
    Outputs: (batch, seq_len, embed_dim) for sequences.
    """
    MAX_SEQ_LEN = 20
    
    if embed_type == 'TF-IDF':
        # TF-IDF must be fitted fresh to avoid data leakage
        vec = TfidfVectorizer(max_features=5000)
        X_train = vec.fit_transform(texts_train).toarray()
        X_test = vec.transform(texts_test).toarray()
        # Add sequence dimension for CNN/LSTM
        return np.expand_dims(X_train, 1), np.expand_dims(X_test, 1), X_train.shape[1]
        
    elif embed_type == 'GloVe':
        global _GLOVE
        if _GLOVE is None:
            _GLOVE = api.load("glove-wiki-gigaword-300")
            
        def embed_glove(texts):
            X = np.zeros((len(texts), MAX_SEQ_LEN, 300), dtype=np.float32)
            for i, text in enumerate(texts):
                if text in _TEXT_CACHE['GloVe']:
                    X[i] = _TEXT_CACHE['GloVe'][text]
                else:
                    tokens = simple_tokenize(text)[:MAX_SEQ_LEN]
                    for j, token in enumerate(tokens):
                        if token in _GLOVE:
                            X[i, j, :] = _GLOVE[token]
                    _TEXT_CACHE['GloVe'][text] = X[i].copy()
            return X
            
        return embed_glove(texts_train), embed_glove(texts_test), 300
        
    elif embed_type == 'Word2Vec':
        # W2V must be fitted fresh on train set to avoid data leakage
        tokenized_train = [simple_tokenize(t) for t in texts_train]
        w2v = Word2Vec(sentences=tokenized_train, vector_size=300, window=5, min_count=1, workers=1)
        
        def embed_w2v(texts):
            X = np.zeros((len(texts), MAX_SEQ_LEN, 300), dtype=np.float32)
            for i, text in enumerate(texts):
                tokens = simple_tokenize(text)[:MAX_SEQ_LEN]
                for j, token in enumerate(tokens):
                    if token in w2v.wv:
                        X[i, j, :] = w2v.wv[token]
            return X
            
        return embed_w2v(texts_train), embed_w2v(texts_test), 300
        
    elif embed_type == 'BERT':
        global _BERT_MODEL, _BERT_TOK
        if _BERT_MODEL is None:
            _BERT_TOK = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            _BERT_MODEL = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            _BERT_MODEL.eval()
            torch.set_num_threads(1)
            
        def embed_bert(texts):
            X = np.zeros((len(texts), MAX_SEQ_LEN, 384), dtype=np.float32)
            uncached_idx = []
            uncached_texts = []
            for i, text in enumerate(texts):
                if text in _TEXT_CACHE['BERT']:
                    X[i] = _TEXT_CACHE['BERT'][text]
                else:
                    uncached_idx.append(i)
                    uncached_texts.append(text)
            
            if uncached_texts:
                batch_size = 64
                texts_str = [str(t) for t in uncached_texts]
                for i in range(0, len(texts_str), batch_size):
                    batch = texts_str[i:i+batch_size]
                    encoded = _BERT_TOK(batch, padding='max_length', truncation=True, max_length=MAX_SEQ_LEN, return_tensors='pt')
                    with torch.no_grad():
                        outputs = _BERT_MODEL(**encoded)
                        hidden = outputs.last_hidden_state.numpy()
                    
                    for j, h in enumerate(hidden):
                        global_i = uncached_idx[i + j]
                        X[global_i] = h
                        _TEXT_CACHE['BERT'][uncached_texts[i + j]] = h.copy()
            return X
            
        return embed_bert(texts_train), embed_bert(texts_test), 384

    elif embed_type == 'SBERT':
        global _SBERT_MODEL
        if _SBERT_MODEL is None:
            from sentence_transformers import SentenceTransformer
            _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            
        def embed_sbert(texts):
            X = np.zeros((len(texts), 1, 384), dtype=np.float32)
            uncached_idx = []
            uncached_texts = []
            for i, text in enumerate(texts):
                if text in _TEXT_CACHE['SBERT']:
                    X[i] = _TEXT_CACHE['SBERT'][text]
                else:
                    uncached_idx.append(i)
                    uncached_texts.append(text)
                    
            if uncached_texts:
                new_embs = _SBERT_MODEL.encode(uncached_texts, show_progress_bar=False)
                new_embs = np.expand_dims(new_embs, axis=1)
                for idx_local, global_idx in enumerate(uncached_idx):
                    X[global_idx] = new_embs[idx_local]
                    _TEXT_CACHE['SBERT'][uncached_texts[idx_local]] = new_embs[idx_local].copy()
            return X
        return embed_sbert(texts_train), embed_sbert(texts_test), 384
        
    elif embed_type == 'MPNet':
        global _MPNET_MODEL
        if _MPNET_MODEL is None:
            from sentence_transformers import SentenceTransformer
            _MPNET_MODEL = SentenceTransformer("all-mpnet-base-v2")
            
        def embed_mpnet(texts):
            X = np.zeros((len(texts), 1, 768), dtype=np.float32)
            uncached_idx = []
            uncached_texts = []
            for i, text in enumerate(texts):
                if text in _TEXT_CACHE['MPNet']:
                    X[i] = _TEXT_CACHE['MPNet'][text]
                else:
                    uncached_idx.append(i)
                    uncached_texts.append(text)
                    
            if uncached_texts:
                new_embs = _MPNET_MODEL.encode(uncached_texts, show_progress_bar=False)
                new_embs = np.expand_dims(new_embs, axis=1)
                for idx_local, global_idx in enumerate(uncached_idx):
                    X[global_idx] = new_embs[idx_local]
                    _TEXT_CACHE['MPNet'][uncached_texts[idx_local]] = new_embs[idx_local].copy()
            return X
        return embed_mpnet(texts_train), embed_mpnet(texts_test), 768

def get_hybrid_embeddings(texts_train, texts_test, combination: tuple):
    embs_data = []
    
    for embed in combination:
        embs_data.append(get_deep_embeddings(texts_train, texts_test, embed))
        
    max_seq_len = 1
    for tr, _, _ in embs_data:
        if tr.shape[1] > max_seq_len:
            max_seq_len = tr.shape[1]
            
    train_embs = []
    test_embs = []
    total_dim = 0
    
    for tr, te, dim in embs_data:
        tr = tr.astype(np.float32)
        te = te.astype(np.float32)
        
        if tr.shape[1] < max_seq_len:
            tr = np.repeat(tr, max_seq_len, axis=1)
            te = np.repeat(te, max_seq_len, axis=1)
            
        train_embs.append(tr)
        test_embs.append(te)
        total_dim += dim
        
    X_train_hybrid = np.concatenate(train_embs, axis=2)
    X_test_hybrid = np.concatenate(test_embs, axis=2)
    
    return X_train_hybrid, X_test_hybrid, X_train_hybrid.shape[2]
