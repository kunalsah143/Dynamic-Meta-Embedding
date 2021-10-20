# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 20:51:16 2021

@author: KUNAL
"""

import numpy as np
import matplotlib.pyplot as plt
import  gensim
from gensim.models import KeyedVectors
#w2v = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
#%%
from gensim.models import Word2Vec
from gensim.models import FastText
import nltk
nltk.download('punkt')
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import string
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

#%%
glove = {}
with open("glove.6B.100d.txt", 'r', encoding="utf-8") as f:
  for line in f:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], "float32")
    glove[word] = vector
#%%
data_pos = open('rt-polaritydata/rt-polarity.pos').readlines()
data_neg = open('rt-polaritydata/rt-polarity.neg').readlines()

train_d = data_pos[:-831] + data_neg[:-831]
train_l = [1]*(len(data_pos)-831)+[0]*(len(data_neg)-831)
test_d = data_pos[-831:] + data_neg[-831:]
test_l = [1]*(831)+[0]*(831)

temp = list(zip(train_d, train_l))
np.random.shuffle(temp)
train_d, train_l = zip(*temp)

temp = list(zip(test_d, test_l))
np.random.shuffle(temp)
test_d, test_l = zip(*temp)


#%%
# process sentences to tokens
text_p = ["".join([char for char in line if char not in string.punctuation]) for line in train_d]
lines = [word_tokenize(text) for text in text_p]
train_data = [[word for word in words if len(word)>=2 and len(word)<15] for words in lines]
#create word list from token using utf8 encoding 
#word_list = [word for words in processedLines for word in words]

text_p = ["".join([char for char in line if char not in string.punctuation]) for line in test_d]
lines = [word_tokenize(text) for text in text_p]
test_data = [[word for word in words if len(word)>=2 and len(word)<15] for words in lines]

#%%
'''w2v = Word2Vec(vector_size=100, min_count=1)
w2v.build_vocab(train_data)
total_examples = w2v.corpus_count
#w2v_2.build_vocab([w2v.index_to_key], update=True)
#w2v_2.intersect_word2vec_format(".GoogleNews-vectors-negative300.bin", binary=True)
w2v.train(train_data, total_examples=total_examples, epochs=1000)'''
w2v = Word2Vec(train_data, workers = 4, vector_size = 100, epochs = 100, window=5, min_count = 1)
w2v.save('word2vec.model')
#%%

ftxt = FastText(vector_size= 100, window = 5,sentences = train_data,epochs = 100)
ftxt.save('fasttext.model')

#%%

w2v = Word2Vec.load('word2vec.model')
ftxt = FastText.load('fasttext.model')

print(w2v.wv['music'])

plt.title('sequence lengths in train data')
plt.bar(range(len(train_data)),[len(t) for t in train_data])
plt.show()
#%%
X_train_w2v = []
X_test_w2v = []
X_train_glove = []
X_test_glove = []
X_train_ftxt = []
X_test_ftxt = []
pad = np.zeros(100)
oov = np.zeros(100)
seq_len = 40

for i in range(len(train_data)):
    t = []
    t1 = []
    t2 = []
    n = min(len(train_data[i]),seq_len)
    for j  in range(n):
        try:
            t.append(list(glove[train_data[i][j]]))
        except:
            t.append(oov)
        try:
            t1.append(list(w2v.wv[train_data[i][j]]))
        except:
            t1.append(oov)
        try:
            t2.append(list(ftxt.wv[train_data[i][j]]))
        except:
            t2.append(oov)
    for k in range(n,seq_len):
        t.append(pad)
        t1.append(pad)
        t2.append(pad)
    X_train_glove.append(t)
    X_train_w2v.append(t1)
    X_train_ftxt.append(t2)
    
X_train_glove = np.array(X_train_glove)
X_train_w2v = np.array(X_train_w2v)
X_train_ftxt = np.array(X_train_ftxt)
y_train = np.array(train_l)

#%%
for i in range(len(test_data)):
    t = []
    t1 = []
    t2 = []
    n = min(len(test_data[i]),seq_len)
    for j  in range(n):
        try:
            t.append(list(glove[test_data[i][j]]))
        except:
            t.append(oov)
        try:
            t1.append(list(w2v.wv[test_data[i][j]]))
        except:
            t1.append(oov)
        try:
            t2.append(list(ftxt.wv[test_data[i][j]]))
        except:
            t2.append(oov)
    for k in range(n,seq_len):
        t.append(pad)
        t1.append(pad)
        t2.append(pad)
    X_test_glove.append(t)
    X_test_w2v.append(t1)
    X_test_ftxt.append(t2)
    
X_test_glove = np.array(X_test_glove)
X_test_w2v = np.array(X_test_w2v)
X_test_ftxt = np.array(X_test_ftxt)
y_test = np.array(test_l)


#%%
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class DME(nn.Module):
    
    def __init__(self,input_dim = 100, hidden_dim = 50):
        super(DME, self).__init__()
        self.w2v_fc1 = nn.Linear(input_dim,hidden_dim)
        self.ftxt_fc1 = nn.Linear(input_dim,hidden_dim)
        self.glove_fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self,x_w,x_f,x_g):
        x_w2v = self.w2v_fc1(x_w)
        x_ftxt = self.ftxt_fc1(x_f)
        x_glove = self.glove_fc1(x_g)
        x_w2v = self.fc2(x_w2v)
        x_ftxt = self.fc2(x_ftxt)
        x_glove = self.fc2(x_glove)
        #print(x_w2v.shape,x_ftxt.shape)
        t = torch.cat((x_w2v, x_ftxt, x_glove),2)
        #print(t.shape)
        return self.softmax(t)

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout = dropout,batch_first = True)
        
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        
        outputs, (hidden, cell) = self.rnn(src)
        '''h1 = (hidden[0,:,:] + hidden[1,:,:])/2
        h2 =  (hidden[2,:,:] + hidden[3,:,:])/2
        h = torch.stack((h1,h2),dim=0)
        
        c1 = (cell[0,:,:] + cell[1,:,:])/2
        c2 =  (cell[2,:,:] + cell[3,:,:])/2
        c = torch.stack((c1,c2),dim=0)'''
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        return outputs[:,-1,:]
class Decoder(nn.Module):
    def __init__(self,input_dim, output_dim):
        super().__init__()
        
        self.output_dim = output_dim
        
        
        self.fc_out = nn.Linear(input_dim, 10)
        self.fc_out1 = nn.Linear(10,output_dim)
        self.act = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        
    def forward(self, input):
        
        prediction = self.fc_out1(self.fc_out(input))
        
        
        #prediction = [batch size, output dim]
        
        return prediction
    
class reviewDetection(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        
    def forward(self, src):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_out = self.encoder(src)
        #first input to the decoder is the <sos> tokens
        
        output = self.decoder(enc_out)
        
        return output
    

#%%%

INPUT_DIM = 100
OUTPUT_DIM = 2
HID_DIM = 128
N_LAYERS = 1
ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2

batch_size = 128
size = X_train_w2v.shape[0]
d = int(size/batch_size)
val_size = int(0.2*X_train_w2v.shape[0])
X_val_w2v = X_train_w2v[:val_size,:,:]
X_val_ftxt = X_train_ftxt[:val_size,:,:]
X_val_glove = X_train_glove[:val_size,:,:]
y_val = y_train[:val_size]
X_train_w2v_b = np.array_split(X_train_w2v[val_size:,:,:],d)
X_train_ftxt_b = np.array_split(X_train_ftxt[val_size:,:,:],d)
X_train_glove_b = np.array_split(X_train_glove[val_size:,:,:],d)
y_train_b = np.array_split(y_train[val_size:],d)

enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(HID_DIM, OUTPUT_DIM)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.0004

model = reviewDetection(enc, dec, device).to(device)
metaE = DME().to(device)
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
#model.apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters(),lr=lr)


criterion = nn.CrossEntropyLoss()

def train(metaE,model, X_train_w2v, X_train_ftxt, X_train_glove, y_train, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    n_samples = 0
    corr = 0

    for X_w,X_f,X_g,y in zip(X_train_w2v,X_train_ftxt,X_train_glove,y_train):
        
        n_samples += X_w.shape[0]
        X_w = torch.from_numpy(X_w.astype(np.float32))
        X_w = X_w.to(device)
        X_f = torch.from_numpy(X_f.astype(np.float32))
        X_f = X_f.to(device)
        X_g = torch.from_numpy(X_g.astype(np.float32))
        X_g = X_g.to(device)
        
        
        dm = metaE(X_w,X_f,X_g)
        
        X  = dm[:,:,0:1]*X_w + dm[:,:,1:2]*X_f + dm[:,:,2:]*X_g
        
        y = torch.from_numpy(y)
        y = y.to(device)
        optimizer.zero_grad()
        
        output = model(X)
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        loss = criterion(output, y.long())
        
        loss.backward()
        
        #torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()*X.shape[0]
        y_pred = np.argmax(output.detach().cpu().numpy(),axis=1)
        corr += np.sum(y_pred==y.detach().cpu().numpy())
        #print(output,y,corr)
    return epoch_loss / n_samples, corr/n_samples

def evaluate(metaE, model, X_test_w2v, X_test_ftxt, X_test_glove, y_test, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
        
        n_samples = X_test_w2v.shape[0]
        X_w = torch.from_numpy(X_test_w2v.astype(np.float32))
        X_w = X_w.to(device)
        X_f = torch.from_numpy(X_test_ftxt.astype(np.float32))
        X_f = X_f.to(device)
        X_g = torch.from_numpy(X_test_glove.astype(np.float32))
        X_g = X_g.to(device)
        
        
        dm = metaE(X_w,X_f,X_g)
        
        X  = dm[:,:,0:1]*X_w + dm[:,:,1:2]*X_f + dm[:,:,2:]*X_g
        
        y = torch.from_numpy(y_test)
        y = y.to(device)
        optimizer.zero_grad()
        
        output = model(X)
        

        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, y.long())
        
        epoch_loss += loss.item()
        
    return epoch_loss,output

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 50
CLIP = 1
val_losses = []
train_losses = []
val_accs = []
train_accs = []
best_valid_acc = 0
for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss,train_acc = train(metaE, model, X_train_w2v_b, X_train_ftxt_b, X_train_glove_b, y_train_b, optimizer, criterion, CLIP)
    valid_loss,y_pred = evaluate(metaE, model, X_val_w2v, X_val_ftxt, X_val_glove, y_val, criterion)
    y_pred = np.argmax(y_pred.detach().cpu().numpy(),axis=1)
    valid_acc = np.mean(y_val==y_pred)
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model, 'model_meta1.sav')
        torch.save(metaE,'metaE.sav')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} Acc : {train_acc}')
    print(f'\t Val. Loss: {valid_loss:.3f} Acc : {np.mean(y_val==y_pred)}')
    val_losses.append(valid_loss)
    val_accs.append(valid_acc)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
#%%
model = torch.load('model_meta.sav')
metaE = torch.load('metaE.sav')
test_loss,y_pred = evaluate(metaE, model, X_test_w2v, X_test_ftxt, X_test_glove, y_test, criterion)
y_pred = np.argmax(y_pred.detach().cpu().numpy(),axis=1)
print(f'Test. Loss: {test_loss:.3f} Acc : {np.mean(y_test==y_pred)}',)
 #%%
 plt.title('Loss vs Epochs')
 plt.plot(val_losses,label='valid')
 plt.plot(train_losses,label='train')
 plt.legend()
 plt.show()
 plt.plot(val_accs,label='valid_acc')
 plt.plot(train_accs,label='train_acc')
 plt.legend()
 plt.show()
 
 #%%
