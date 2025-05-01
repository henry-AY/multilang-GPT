import os
import csv
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

import config
from model import BigramLanguageModel

torch.manual_seed(1)

with open(config.data, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch : i for i, ch in enumerate(chars) }
itos = {i : ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, outut a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will train, rest validate
train_data = data[:n]
val_data = data[n:]

train_data[:config.block_size + 1] #temp?

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i : i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + config.block_size + 1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Function to save checkpoint
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved at epoch {epoch}.")

# Function to load checkpoint
def load_checkpoint(path, model, optimizer):
    if os.path.exists(path):
        checkpoint = torch.load(path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded: Resuming from epoch {epoch} with loss {loss:.4f}")
        return epoch, loss
    else:
        print("No checkpoint found. Training from scratch.")
        return 0, None

checkpoint_path = config.BASE_DIR.parent / 'model' / 'checkpoint.pth'
logs_path = config.BASE_DIR.parent / 'output' / 'datalogs' / 'training_log.csv'

model = BigramLanguageModel(vocab_size).to(config.device)
optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate)
model.train()

def main():
    # Load checkpoint if exists
    start_epoch, start_loss = load_checkpoint(checkpoint_path, model, optimizer)

    user_epochs = int(input(f"Current epoch is {start_epoch}. How many more epochs do you want to train?"))
    num_epochs = start_epoch + user_epochs  # Compute total epochs

    csvfile = open(logs_path, "w", newline='') #open outside of with statement
    fieldnames = ['step', 'train_loss', 'val_loss']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for epoch in range(start_epoch, num_epochs):
        for iter in range(config.max_iters):
            if iter % config.eval_interval == 0:
                losses = estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")  # Write to console
                writer.writerow({
                    'step': iter,
                    'train_loss': losses['train'].item(),
                    'val_loss': losses['val'].item()
                })  # Write to CSV
            #sample a batch of data
            xb, yb = get_batch('train')

            #evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none = True)
            loss.backward()
            optimizer.step()

        save_checkpoint(model, optimizer, epoch, loss.item(), checkpoint_path)

        losses = estimate_loss()
        print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    model_state = config.BASE_DIR.parent / 'model' / 'final_model_weights.pth'
    optimizer_state = config.BASE_DIR.parent / 'model' / 'final_model_optimizer.pth'


    torch.save(model.state_dict(), model_state)
    torch.save(optimizer.state_dict(), optimizer_state)

if __name__ == "__main__":
    main()