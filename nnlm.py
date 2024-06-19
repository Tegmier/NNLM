import torch
import torch.nn as nn
import pickle 
import tool
from collections import Counter

# data_loading
with open(r'data/corpus.pkl', 'rb') as f:
    data = pickle.load(f)

with open(r'data/voc.pkl', 'rb') as f:
    voc = pickle.load(f)

word_to_index = {word: idx + 1 for idx, word in enumerate(voc)}
word_to_index['<PAD>'] = 0



# word_to_one_hot = {word: tool.word_to_one_hot(word, word_to_index) for word in voc}

# word_to_one_hot = torch.tensor(word_to_one_hot, dtype = torch.int64)

sequence, data = tool.sentence_padding(data)

for sentence in data:
    for i in range(len(sentence)):
        sentence[i] = word_to_index[sentence[i]]
print(data)
word_to_one_hot = []
# for word, index in word_to_index.items():
#     word_to_one_hot.append(tool.word_to_one_hot(word, word_to_index))

batch_size = 1
win_size = 5
# 169
voc_size = len(voc)
# 169
embedding_size = word_to_one_hot[0].size(0)

class Model(nn.Module):
    def __init__(self, batch_size, win_size, voc_size) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.win_size = win_size
        self.voc_size = voc_size
        self.layer1 = nn.Linear(embedding_size * win_size, voc_size)
        self.layer2 = nn.Linear(embedding_size * voc_size, voc_size)
    




