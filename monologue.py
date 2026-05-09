# Nobody is listening. Generate anyway. 
import os
import pickle
import torch

from stranger import GPT, GPTConfig

out_dir = 'checkpoints'
data_dir = 'data/shakespeare_char'
ckpt_path = os.path.join(out_dir, 'ckpt.pt')

prompt = "ROMEO: "
max_new_tokens = 500
temperature = 0.8
top_k = 200
num_samples = 3
seed = 1337

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(seed)

#This part loads vocab for encoding/decoding
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
def encode(s):
    return[stoi[c] for c in s]
def decode(l):
    return ''.join([itos[i] for i in l])

#load full checkpoint, not just weights because this checkpoint includes things like config, iter_num, and best_val_loss
print(f"loading checkpoint from {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location = device, weights_only = False)
print(f" trained for {ckpt['iter_num']} iters, best val loss {ckpt['best_val_loss']:.4f}")

#reconstructs model from save config and load weights
config = ckpt['config']
model = GPT(config)
model.load_state_dict(ckpt['model'])
model.eval()
model.to(device)

#encode prompt as Token IDs, shape (1, T) - batch size 1, sequence length T
prompt_ids = encode(prompt)
x = torch.tensor(prompt_ids, dtype = torch.long, device = device)[None, ...]

print()
print(f"prompt: {prompt!r}")
print(f"generating {num_samples} samples of {max_new_tokens} tokens (temp = {temperature}, top_k = {top_k})")
print("=" * 60)

for i in range(num_samples):
    out_ids = model.generate(x, max_new_tokens = max_new_tokens, temperature = temperature, top_k = top_k)
    text = decode(out_ids[0].tolist())
    print(f"\n--- sample {i+1} ---")
    print(text)
    print()