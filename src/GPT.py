import random
import sys
import torch
import torch.nn as nn
from model import BigramLanguageModel

import config
from pathlib import Path

with open(config.data, 'r', encoding='utf-8') as f:
    text = f.read()

""" REPLACE WITH SINGLE SEED FOR REPEATABLE RESULTS"""
random_seed = random.randint(0, sys.maxsize - 1)
torch.manual_seed(random_seed)

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch : i for i, ch in enumerate(chars) }
itos = {i : ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, outut a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

state_dict = config.BASE_DIR.parent / 'model' / 'final_model_weights.pth'

model = BigramLanguageModel(vocab_size).to(config.device)
model.load_state_dict(torch.load(state_dict, weights_only=True))
model.eval()

checkpoint_path = config.BASE_DIR.parent / 'model' / 'checkpoint.pth'

checkpoint = torch.load(checkpoint_path, weights_only=True)

curr_epoch = checkpoint['epoch']

def generate_token(token, max_tokens):
    input_ids = torch.tensor([encode(token)], dtype=torch.long).to(config.device)
    generated_ids = model.generate(input_ids, max_new_tokens=max_tokens)
    return (decode(generated_ids[0].tolist()))

output = generate_token("The Prince", max_tokens=500)

#print GPT, and license
print('jh-GPT: Mit license\nAuthors: henry-Ay, jgarc826\nhttps://github.com/henry-AY/multilang-GPT')
print(f'\nRandom seed set to: {random_seed}\nEpoch: {curr_epoch}\n')

output_path = config.BASE_DIR.parent / 'output' / 'text_logs' / 'output.txt'
f = open(output_path, 'a')
f.write(f'\nOutput @ Epoch: {curr_epoch}\n{output}\n')
f.close()

#print to console
print(output)