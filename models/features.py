"""
models/features.py
==================
Feature extraction for all four embedding families used in the paper:
  1. TF-IDF (word 1-3gram + character 3-5gram)   — primary, sparse
  2. GloVe 6B 300d (averaged per-sentence)        — static dense
  3. Word2Vec 300d (corpus-trained, averaged)      — static dense
  4. Sentence-BERT (all-MiniLM-L6-v2 / all-mpnet-base-v2) — contextual dense

Usage pattern (same for all)
-----------------------------
    builder = TFIDFBuilder()
    X_train_feat = builder.fit_transform(X_train)
    X_test_feat  = builder.transform(X_test)
"""

from __future__ import annotations
import os
import re
import warnings
import numpy as np
import ssl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure HuggingFace models can be loaded safely
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# Bypass SSL verification for huggingface downloads on constrained networks
import ssl
import urllib.request
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix

ssl._create_default_https_context = ssl._create_unverified_context
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    import requests
    _orig_send = requests.Session.send
    def _patched_send(self, request, **kwargs):
        kwargs['verify'] = False
        return _orig_send(self, request, **kwargs)
    requests.Session.send = _patched_send
except Exception:
    pass

try:
    import torch
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' in kwargs:
            kwargs['weights_only'] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
except Exception:
    pass

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
warnings.filterwarnings("ignore")


# ===========================================================================
# 1. TF-IDF (word + character n-grams)
# ===========================================================================

class TFIDFBuilder:
    """
    Combined word (1-3gram) + character (3-5gram) TF-IDF representation.

    Equation 1 in the paper:
        TF-IDF(t,d,D) = (1 + log f_{t,d}) * log(|D| / df(t))

    Parameters
    ----------
    word_ngram_range : tuple, default (1, 3)
    char_ngram_range : tuple, default (3, 5)
    min_df           : int,   default 2
    sublinear_tf     : bool,  default True
    """

    def __init__(
        self,
        word_ngram_range: tuple[int, int] = (1, 3),
        char_ngram_range: tuple[int, int] = (3, 5),
        min_df: int = 2,
        sublinear_tf: bool = True,
    ):
        self._union = FeatureUnion([
            ("word", TfidfVectorizer(
                sublinear_tf=sublinear_tf,
                ngram_range=word_ngram_range,
                min_df=min_df,
                stop_words="english",
            )),
            ("char", TfidfVectorizer(
                sublinear_tf=sublinear_tf,
                analyzer="char_wb",
                ngram_range=char_ngram_range,
                min_df=min_df,
            )),
        ])
        self.fitted = False

    def fit_transform(self, texts: np.ndarray) -> csr_matrix:
        A = self._union.fit_transform(texts)
        self.fitted = True
        return A

    def transform(self, texts: np.ndarray) -> csr_matrix:
        if not self.fitted:
            raise RuntimeError("Call fit_transform first.")
        return self._union.transform(texts)

    def vocab_size(self) -> int:
        return sum(
            len(t.vocabulary_)
            for _, t in self._union.transformer_list
        )


# ===========================================================================
# 2. GloVe (averaged token vectors)
# ===========================================================================

class GloVeBuilder:
    """
    Averaged GloVe 6B 300d sentence representation.
    Downloads directly from Stanford site and caches locally.
    """

    # No longer uses gensim; vectors are loaded from a text file.
    _cache = {}
    _kv = None

    def __init__(self):
        pass

    def _load(self) -> None:
        if self._kv is not None:
            return
        import os
        import pathlib
        import urllib.request
        import zipfile
        import numpy as np
        import re

        # Directory to store downloaded GloVe files
        cache_dir = pathlib.Path(__file__).parents[2] / "data" / "glove"
        cache_dir.mkdir(parents=True, exist_ok=True)
        glove_txt = cache_dir / "glove.6B.300d.txt"
        if not glove_txt.is_file():
            zip_path = cache_dir / "glove.6B.zip"
            if not zip_path.is_file():
                print("  Downloading GloVe zip (approx 822 MB) - this may take several minutes...")
                url = "https://nlp.stanford.edu/data/glove.6B.zip"
                urllib.request.urlretrieve(url, zip_path)
            print("  Extracting glove.6B.300d.txt ...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extract("glove.6B.300d.txt", path=cache_dir)
        print(f"  Loading GloVe vectors from {glove_txt} ...")
        kv = {}
        with open(glove_txt, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                vec = np.array(list(map(float, parts[1:])), dtype="float32")
                kv[word] = vec
        self._kv = kv

    def _encode_one(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]
        tokens = re.findall(r"[a-z]+", str(text).lower())
        vecs = [self._kv[w] for w in tokens if w in self._kv]
        if vecs:
            emb = np.mean(vecs, axis=0).astype("float32")
        else:
            emb = np.zeros(300, dtype="float32")
        self._cache[text] = emb
        return emb

    def fit_transform(self, texts: np.ndarray) -> np.ndarray:
        self._load()
        return np.vstack([self._encode_one(t) for t in texts])

    def transform(self, texts: np.ndarray) -> np.ndarray:
        self._load()
        return np.vstack([self._encode_one(t) for t in texts])


# ===========================================================================
# 3. Word2Vec (corpus-trained, averaged)
# ===========================================================================

class Word2VecBuilder:
    """
    Corpus-trained Word2Vec skip-gram (300d, averaged).
    Must be fitted on training data before transform.
    """

    def __init__(self, vector_size: int = 300, window: int = 5,
                 min_count: int = 2, epochs: int = 30, seed: int = 42):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed
        self._model = None

    @staticmethod
    def _tokenise(texts: np.ndarray) -> list[list[str]]:
        return [re.findall(r"[a-z]+", str(t).lower()) for t in texts]

    def _encode_one(self, tokens: list[str]) -> np.ndarray:
        vecs = [self._model.wv[w] for w in tokens if w in self._model.wv]
        if vecs:
            return np.mean(vecs, axis=0).astype("float32")
        return np.zeros(self.vector_size, dtype="float32")

    def fit_transform(self, texts: np.ndarray) -> np.ndarray:
        from gensim.models import Word2Vec
        sents = self._tokenise(texts)
        self._model = Word2Vec(
            sents, vector_size=self.vector_size, window=self.window,
            min_count=self.min_count, workers=1, seed=self.seed,
            epochs=self.epochs,
        )
        return np.vstack([self._encode_one(s) for s in sents])

    def transform(self, texts: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit_transform first.")
        return np.vstack([self._encode_one(self._tokenise([t])[0]) for t in texts])


# ===========================================================================
# 4. Sentence-BERT
# ===========================================================================

class FeatureBuilder:
    pass

class SBERTBuilder(FeatureBuilder):
    """
    Computes dense embeddings using SentenceTransformers (MiniLM).
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.batch_size = 32
        
    def _encode(self, texts):
        if self._model is None:
            # Lazy load to save memory
            from sentence_transformers import SentenceTransformer
            print(f"  Loading sentence-transformer ({self.model_name})...")
            self._model = SentenceTransformer(self.model_name)
            
        unique_texts = list(set(texts))
        
        # We process in batches to avoid OOM
        # But SentenceTransformer.encode already does batching
        encoded_missing = self._model.encode(
            unique_texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        
        embed_dict = dict(zip(unique_texts, encoded_missing))
        return np.array([embed_dict[t] for t in texts], dtype=np.float32)

    def fit_transform(self, texts):
        return self._encode(texts)

    def transform(self, texts):
        return self._encode(texts)

class CustomBERTBuilder(FeatureBuilder):
    """
    Wraps standard HuggingFace BERT models (like seBERT) into Sentence-Transformers.
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self._model = None
        self.batch_size = 16 # smaller batch size for full-sized BERTs
        
    def _encode(self, texts):
        if self._model is None:
            # Lazy load to save memory
            from sentence_transformers import SentenceTransformer, models
            print(f"  Loading custom domain BERT ({self.model_name})...")
            word_embedding_model = models.Transformer(self.model_name)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            self._model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            
        unique_texts = list(set(texts))
        
        encoded_missing = self._model.encode(
            unique_texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        
        embed_dict = dict(zip(unique_texts, encoded_missing))
        return np.array([embed_dict[t] for t in texts], dtype=np.float32)

    def fit_transform(self, texts):
        return self._encode(texts)

    def transform(self, texts):
        return self._encode(texts)

class SEBERTBuilder(CustomBERTBuilder):
    def __init__(self):
        super().__init__(model_name="thearod5/se-bert")

class REBERTBuilder(CustomBERTBuilder):
    def __init__(self):
        super().__init__(model_name="Jingye/BERT4RE")


# ===========================================================================
# 5. Hybrid: TF-IDF + dense embedding (concatenated)
# ===========================================================================

class HybridBuilder:
    """
    Concatenate sparse TF-IDF with a dense sentence embedding.
    Used as the winning configuration for PROMISE_exp.
    """

    def __init__(
        self,
        dense_builder: SBERTBuilder | GloVeBuilder | Word2VecBuilder,
        tfidf_params: dict | None = None,
    ):
        self._tfidf = TFIDFBuilder(**(tfidf_params or {}))
        self._dense = dense_builder

    def fit_transform(self, texts: np.ndarray):
        A = self._tfidf.fit_transform(texts)
        E = self._dense.fit_transform(texts)
        return hstack([A, csr_matrix(E)]).tocsr()

    def transform(self, texts: np.ndarray):
        A = self._tfidf.transform(texts)
        E = self._dense.transform(texts)
        return hstack([A, csr_matrix(E)]).tocsr()


if __name__ == "__main__":
    texts = np.array([
        "the system shall respond within three seconds",
        "the software shall be available 99.9 percent of the time",
        "users shall be able to log in using their email and password",
    ])
    b = TFIDFBuilder()
    X = b.fit_transform(texts)
    print(f"TF-IDF shape: {X.shape}  (sparse)")
