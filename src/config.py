# config.py

import torch.nn

#hyperparams
batch_size = 32 #how many independent sequences will be processed in parallel
block_size = 8 # what is the maxmimum context length for predictions
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2 # 20% of all intermediate calculations are dropped to 0
learning_rate = 1e-3

max_iters = 1000
eval_interval = 200
eval_iters = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'