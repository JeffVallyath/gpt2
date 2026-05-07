# One must imagine the optimizer happy.
import os
import math
import pickle
import numpy as np
import torch
from stranger import GPT, GPTConfig
import time

out_dir = 'checkpoints'
data_dir = 'data/shakespeare_char'

batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2 #moderate dropout just to prevent overfitting
bias = False

learning_rate = 1e-3
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0


warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4

eval_interval = 250
eval_iters = 200
log_interval = 10

seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(seed)
os.makedirs(out_dir, exist_ok = True)

with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
print(f"vocab_size from meta.pkl: {vocab_size}")

#memmap the whole thing so we dont load the whole thing into RAM
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype = np.uint16, mode = 'r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype = np.uint16, mode = "r")

#turns a long stream of token ids into random input-target training windows
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1: i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

config = GPTConfig(block_size = block_size, vocab_size = vocab_size, n_layer = n_layer, n_head = n_head, n_embd = n_embd, dropout = dropout, bias = bias,)
model = GPT(config)
model.to(device)

#counts total number of trainable scalar values in the model
print(f"model has {sum(p.numel() for  p in model.parameters()):,} parameters")
print(f"training on device: {device}")

#tensors with weight 2 or more are usually weight matrices which get weight decay, otherwise most likely biases or LayerNorm scale parameters, so no weight decay
decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad]
nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2 and p.requires_grad]
optim_groups = [{'params': decay_params, 'weight_decay': weight_decay}, {'params': nodecay_params, 'weight_decay': 0.0}]

optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = (beta1, beta2))

#At the beginning, during warmup, it increases linearly from small to full learning rate. That helps avoid unstable early training. After warmup, it decays smoothly using cosine decay.
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1)/(warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
#how well is the model doing on both seen and unseen data?
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            #store the scalar loss value into the losses tensor, .item() converts 0 dimensional pytorch tensor into a normal number
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    #switch back into training mode after evaluation is done becuase dropout and other training-specific behavior should be active for actual training steps
    model.train()
    #return the dictionary containing average train and validation losses
    return out

best_val_loss = float('inf')
t0 = time.time()

for iter_num in range(max_iters + 1):
    #set learning rate for this iteration
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    #periodic eval and checkpointing
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step{iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6f}")
        #if current validation loss is better than any previous one, and if this isnt just step 0, save checkpoint
        if losses['val'] < best_val_loss and iter_num > 0:
            best_val_loss = losses['val']
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config, 'iter_num': iter_num, 'best_val_loss': best_val_loss,}
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            print(f" saved checkpoint to {out_dir}/ckpt.pt (val {best_val_loss:.4f})")
    #prevents loop from running one more training step after reaching max_iters (final evaluation)
    if iter_num == max_iters:
        break

    #one training step
    x, y = get_batch('train')
    _, loss = model(x, y)
    #clear old gradients before backpropogating the new ones, setnone to True more efficient than zero-ing out tensors
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    if iter_num %  log_interval == 0 and iter_num > 0:
        dt = time.time() - t0
        t0 = time.time()
        print(f"iter {iter_num}: loss {loss.item():.4f}, {dt * 1000 / log_interval:.1f}ms/iter")

