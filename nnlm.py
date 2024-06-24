import torch
import torch.nn as nn
import pickle 
import tool
from collections import Counter
import torch.optim as optim


###################### Tegmier Standard Gpu Checking Processing ######################
# work_place lab:0 home:1 laptop:2
work_place = 1
gpu_setup_ascii_art_start = '''
__________                    .__                 ___________                   .__                 __________________ ____ ___    _________       __                
\______   \__ __  ____   ____ |__| ____    ____   \__    ___/___   ____   _____ |__| ___________   /  _____/\______   \    |   \  /   _____/ _____/  |_ __ ________  
 |       _/  |  \/    \ /    \|  |/    \  / ___\    |    |_/ __ \ / ___\ /     \|  |/ __ \_  __ \ /   \  ___ |     ___/    |   /  \_____  \_/ __ \   __\  |  \____ \ 
 |    |   \  |  /   |  \   |  \  |   |  \/ /_/  >   |    |\  ___// /_/  >  Y Y  \  \  ___/|  | \/ \    \_\  \|    |   |    |  /   /        \  ___/|  | |  |  /  |_> >
 |____|_  /____/|___|  /___|  /__|___|  /\___  /    |____| \___  >___  /|__|_|  /__|\___  >__|     \______  /|____|   |______/   /_______  /\___  >__| |____/|   __/ 
        \/           \/     \/        \//_____/                \/_____/       \/        \/                \/                             \/     \/           |__|    
'''
print(gpu_setup_ascii_art_start)
if work_place == 0:
    torch.cuda.set_device(0)
elif work_place == 1:
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    torch.cuda.set_device(0)
else:
    torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.cuda.current_device()
if torch.cuda.is_available() and device != 'cpu':
    print(f"当前设备: CUDA")
    print(f"设备名称: {torch.cuda.get_device_name(device)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(device)}")
    print(f"总内存: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.2f} GB")
else:
    print("当前设备: CPU")

gpu_setup_ascii_art_end = '''
  __________________ ____ ___    _________       __                 ___________           .___      
 /  _____/\______   \    |   \  /   _____/ _____/  |_ __ ________   \_   _____/ ____    __| _/______
/   \  ___ |     ___/    |   /  \_____  \_/ __ \   __\  |  \____ \   |    __)_ /    \  / __ |/  ___/
\    \_\  \|    |   |    |  /   /        \  ___/|  | |  |  /  |_> >  |        \   |  \/ /_/ |\___ \ 
 \______  /|____|   |______/   /_______  /\___  >__| |____/|   __/  /_______  /___|  /\____ /____  >
        \/                             \/     \/           |__|             \/     \/      \/    \/ 
'''
print(gpu_setup_ascii_art_end)
###################### Tegmier Standard Gpu Checking Processing END ######################

# data_loading
with open(r'data/corpus.pkl', 'rb') as f:
    data = pickle.load(f)

with open(r'data/voc.pkl', 'rb') as f:
    voc = pickle.load(f)

word_to_index = {word: idx + 1 for idx, word in enumerate(voc)}
word_to_index['<PAD>'] = 0

sequence, data = tool.sentence_padding(data)

for sentence in data:
    for i in range(len(sentence)):
        sentence[i] = word_to_index[sentence[i]]
word_to_one_hot = []

for word, index in word_to_index.items():
    word_to_one_hot.append(tool.word_to_one_hot(word, word_to_index))

nepochs = 10
batch_size = 1
win_size = 4
voc_size = len(voc) # 169
seq_size = sequence
embedding_size = len(word_to_one_hot[0]) # 169

for segmented_batch in tool.data_loader(data, batch_size, win_size, seq_size):
    lex, label = segmented_batch

class Model(nn.Module):
    def __init__(self, batch_size, win_size, voc_size, seq_size) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.win_size = win_size
        self.voc_size = voc_size
        self.seq_size = seq_size
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(word_to_one_hot, dtype=torch.float64).cuda())
        self.layer1 = nn.Linear(embedding_size * win_size, voc_size)
        self.layer2 = nn.Linear(embedding_size * voc_size, voc_size)
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, x):
        # torch.Size([1, 5, 4])
        x = self.embedding(x)
        print(x.shape)
        y = x.view(1, 5, 4 * 169)

def train_model(model, criterion, optimizer, nepochs, data):
    model.train()
    for epoch in range(nepochs):
        train_loss = []
        train_data_set = tool.data_loader(data, batch_size, win_size, seq_size)
        for train_data in train_data_set:
            lex = train_data[0]
            label = train_data[1]
            lex = lex.cuda()
            label = label.cuda()
            label = label.reshape(-1)
            y_pred = model(lex)
            loss = criterion(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            train_loss.append([float(loss), lex.size(0)])
            optimizer.step()
    return model



model = Model(batch_size = batch_size, win_size = win_size, voc_size = voc_size, seq_size = seq_size)
criterion = nn.CrossEntropyLoss()
optimier = optim.Adam(model.parameters(), lr=1e-3)

model = train_model(model = model, criterion = criterion, optimizer = optimier, nepochs = nepochs, data = data)
