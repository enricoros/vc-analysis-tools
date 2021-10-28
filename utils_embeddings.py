"""
Enrico 2021 - Utils to create embeddings from strings
"""
import numpy as np
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer

# sentence similarity, using USE from TF-Hub (instead of Sentence-Transformers, for instance)

# model caches
_model_use = None
_model_use_large = None
_model_st_mpnet = None


def text_to_embeds_use_fast(text_list):
    global _model_use
    if _model_use is None:
        _model_use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    text_embeddings = _model_use(text_list)
    correlation = np.inner(text_embeddings, text_embeddings)
    return 'use_fast', text_embeddings.numpy(), correlation


def text_to_embeds_use(text_list):
    global _model_use_large
    if _model_use_large is None:
        _model_use_large = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    text_embeddings = _model_use_large(text_list)
    correlation = np.inner(text_embeddings, text_embeddings)
    return 'use', text_embeddings.numpy(), correlation


def text_to_embeds_mpnet(text_list):
    global _model_st_mpnet
    if _model_st_mpnet is None:
        _model_st_mpnet = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    text_embeddings = _model_st_mpnet.encode(text_list)
    correlation = np.inner(text_embeddings, text_embeddings)
    return 'mpnet', text_embeddings, correlation
