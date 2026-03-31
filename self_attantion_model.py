import torch
import numpy as np
import math as m

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data_trf import *
import torch.distributions as dist
import matplotlib.pyplot as plt

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MacOS local gpu device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f'Using device: {device}')

# Self-attantion module:    
class SelfAttention(nn.Module):
    def __init__(self, emb_size, heads):
        super().__init__()

        self.emb_size = emb_size
        self.heads = heads

        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)
        self.multihead = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        b, t, e = x.size()

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        head_dim = self.emb_size//self.heads

        q = q.view(b,t,self.heads, head_dim).transpose(1,2) # (batch_size, heads, time, head_dim)
        k = k.view(b,t,self.heads, head_dim).transpose(1,2) # (batch_size, heads, time, head_dim)
        v = v.view(b,t,self.heads, head_dim).transpose(1,2) # (batch_size, heads, time, head_dim)

        # (batch_size, heads, time, head_dim) @ (batch_size, heads, head_dim, time) = (batch_size, heads, time, time)
        weights = torch.matmul(q,k.transpose(-2,-1))/m.sqrt(head_dim) # (batch_size, heads, time, time)

        # Masking:
        is_, js_ = torch.triu_indices(t,t,offset=1).to(device)
        weights[..., is_,js_] = float(-m.inf)

        attn_scores = F.softmax(weights, dim=-1) # over the last dimention (dim0 = batch, dim1 = query position, dim2 = key pos)
        out = torch.matmul(attn_scores,v)
        out = out.transpose(1, 2).contiguous().view(b, t, e)

        out = self.multihead(out)
        
        return out
        
# Tranformer block: takes up self-attantion
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, heads, hidden_dim = 4, dropout = 0.3):
        super().__init__()

        self.attention = SelfAttention(emb_dim, heads = heads)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.feedfor = nn.Sequential(nn.Linear(emb_dim, emb_dim*hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(emb_dim*hidden_dim, emb_dim))
        
        self.drop = nn.Dropout(dropout)
        
        
    def forward(self, x):
        attn = self.attention(self.norm1(x))
        x = x + self.drop(attn)
        ff = self.feedfor(self.norm2(x))
        out = x + self.drop(ff)
        return out
    
# Positional embedding:
class PositionalEmb(nn.Module):
    def __init__(self, emb_dim, max_seq_len):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_seq_len=max_seq_len
        self.positional_emb = nn.Embedding(max_seq_len, emb_dim)

    def forward(self, x):
        b,t = x.size()
        pos = torch.arange(0,t,device=x.device).unsqueeze(0).expand(b,t) # (b,t)
        out = self.positional_emb(pos) # (b,t,e)
        return out

# Multi-head self-attantion module: takes up transformer block (6 blocks)
class multi_head_trnasformer_self_attn(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len, heads):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len=max_seq_len
        self.heads=heads

        self.embedding = nn.Embedding(vocab_size, embedding_dim) # (b,t,e)
        self.pos_emb = PositionalEmb(embedding_dim, max_seq_len) # (b,t,e)
        self.tblocks = nn.Sequential(*[TransformerBlock(embedding_dim, heads) for _ in range(6)])
        self.linnear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = x.long()
        emb = self.embedding(x)
        pos = self.pos_emb(x)

        x = emb+pos
        x = self.tblocks(x)

        out = self.linnear(x)
        return out # (b,t,vocab_size)

# Loss function: cross-entropy   
def loss(x, y):
    return nn.CrossEntropyLoss()(x, y)

# Batch creation:
def get_batch(data):
    start_i = torch.from_numpy(np.random.choice(np.arange(len(data)-batch_l, dtype=np.int64), batch_size))
    batch = torch.permute(torch.stack([data[start_i+i] for i in range(batch_l+1)],dim=0),(1,0))
    return batch.to(device)

# Samplling from output distribution:
def sample(lnprobs, temperature=1.0):
    if temperature == 0.0:
        return lnprobs.argmax()
    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)
    return cd.sample()

# Model valuation: in bits
def evaluation(model, num_batches=1000):
    model.eval()
    total_bits = 0
    total = 0
    with torch.no_grad():
        for _ in range(num_batches):
            batch = get_batch(test)
            x = batch[:,:-1]
            y = batch[:, -1] # last token (b,)
            pred = model(x)
            pred_last = pred[:,-1,:] # (b,v)
            log_loss = F.log_softmax(pred_last, dim=-1)
            b = y.size(0)
            log_pred = -log_loss[torch.arange(b), y.long()]

            log_pred_bits = log_pred/m.log(2)
            total_bits += log_pred_bits.sum()
            total += y.size(0)
    
    average_log_pr = total_bits/total
    return average_log_pr


# Language generation samling:
def sampling(model, seed_len = 16, sample_len = 150, temp=1.0):
    model.eval()
    start_i = np.random.randint(0, len(test)-seed_len)
    seed = test[start_i:start_i+seed_len].tolist()

    with torch.no_grad():
        for _ in range(sample_len):
            context = torch.tensor(seed[-batch_l:], device=device).unsqueeze(0).long().to(device) # (1,s), (1,s+1)...(1,s+sample_len)

            pred = model(context) # (1, s+sqmple_len, v)
            logits = pred[0, -1, :]  # (v)

            next_tok = sample(logits, temperature=temp)
            
            seed.append(next_tok.item())
    
    out = ''.join([i2c[i] for i in seed])
    return out

# Model training: training, validation, samplling
def train_model(model, num_batches, eval_freq, sampling_freq, lr, warmup, grad_cl):

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

    train_losses, val_losses = [], []
    samples = []
    grad_norms = []
    running_loss = 0

    model.train()

    for itr in range(num_batches):
        if itr < warmup:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (itr+1)/warmup # e.g. itr 4: lr = 0.001*5/500 = 0.00001...itr 499 lr=0.001*500/500=0.001
        # Training:
        x = get_batch(train).to(device)
        inputx = x[:,:-1] 
        targety = x[:, 1:] # (b,t)
        # Forward pass
        optimizer.zero_grad()
        x_pred = model(inputx) # (b,t,v)
        l = loss(x_pred.reshape(-1,model.vocab_size), targety.reshape(-1)) # (6*20,vocab_size), (6*20)
        if (itr+1) % 1000 == 0:
            print(f'iter {itr} loss {l.item()}')
        running_loss += l.item()

        # Backward pass
        l.backward()
        # grad clipping:
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_cl) # all grad magnitude, if > grad_cl ==> 1.0
        grad_norms.append(total_norm.item())

        optimizer.step()

        # Validation:
        if (itr+1) % eval_freq == 0:
            print(f'Itr: {itr}')
            # training loss:
            tr_loss = running_loss/eval_freq
            train_losses.append(tr_loss)
            running_loss=0
            print(f'training loss {tr_loss}')
            # validation loss:
            val_loss = evaluation(model, num_batches=200)
            val_losses.append(val_loss.item())
            print(f'validation loss: {val_loss}')

            model.train()
        
        # samling:
        if (itr+1) % sampling_freq == 0:
            sample = sampling(model)
            samples.append(sample)
            print(f'sampling: {sample}')

            model.train()

    return train_losses, val_losses, samples, grad_norms



(train, test), (i2c, c2i) = load_toy(final=False)

batch_size = 16
batch_l = 256

model = multi_head_trnasformer_self_attn(vocab_size=len(i2c), embedding_dim=300,max_seq_len=batch_l, heads=6)

num_batches=50000
eval_freq=10000
sampling_freq = 10000
lr=0.001
warmup = 2000
grad_cl = 1.0

train_losses, val_losses, samples, grad_norms = train_model(model, num_batches, eval_freq, sampling_freq, lr, warmup, grad_cl)
