# Prepare dataset for training.
import os
import pickle
import urllib.request
import numpy as np

DATA_DIR = os.path.dirname(__file__)

input_file_path = os.path.join(DATA_DIR, 'input.txt')

if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    print(f"downloading {data_url}...")
    urllib.request.urlretrieve(data_url, input_file_path)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"all the unique characters: {''.join(chars)}")
print(f"vocab_size: {vocab_size:,}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

n = len(data)
train_data = data[:int(n * 0.9)] 
val_data = data[int(n * 0.9):]

train_ids = np.array(encode(train_data), dtype = np.uint16)
val_ids = np.array(encode(val_data), dtype = np.uint16)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

train_ids.tofile(os.path.join(DATA_DIR, 'train.bin'))
val_ids.tofile(os.path.join(DATA_DIR, 'val.bin'))

#save the vocab so we can decode generated tokens back to text later
meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi,}

with open(os.path.join(DATA_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
    
print(f"saved train.bin, val.bin, meta.pkl to {DATA_DIR}")