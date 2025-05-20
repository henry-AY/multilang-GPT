import torch.nn
from pathlib import Path

#data path
BASE_DIR = Path(__file__).resolve().parent
data = BASE_DIR.parent / 'data' / 'English' / 'english.txt'

#hyperparams
batch_size = 32 # How many independent sequences will be processed in parallel
block_size = 8 # What is the maxmimum context length for predictions


#transformer params
# Was 384
n_embd = 768 # Dimensionality of the embedding vectors and hidden states
# Was 6
n_head = 12 # Number of attention heads in each multi-head attention layer
# Was 6
n_layer = 12 # Number of transformer blocks (i.e., layers) in the model
dropout = 0.2 # Dropout rate (20%) applied to intermediate layers to reduce overfitting

#optimization params
learning_rate = 1e-3 # Step size used by the optimizer to update weights during training

max_iters = 1000 # Total number of training iterations (batches processed)
eval_interval = 200 # Number of iterations between each model evaluation and logging
eval_iters = 200 # Number of iterations used to estimate evaluation loss (averaged over this many steps)

if torch.cuda.is_available(): # use nvidia cuda if available
    device = torch.device("cuda")
    print("cuda\n")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("mps\n")
else:
    device = torch.device("cpu")