import torch
import torch.nn as nn
import pickle 
import tool

# data_loading
with open(r'data/corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)

voc = set()
for sentence in corpus:
    sentence_ = sentence.strip('.')
    sentence_ = sentence_.lower()
    words = sentence_.split(' ')
    for word in words:
        voc.add(word)

word_to_index = {word: idx for idx, word in enumerate(voc)}

word_to_one_hot = {word: tool.word_to_one_hot(word, word_to_index) for word in voc}

word_to_one_hot = torch.tensor(word_to_one_hot, dtype = torch.int64)

print(word_to_one_hot)

