from rwkv_src.rwkv_tokenizer import RWKV_TOKENIZER, ABCTokenizer
from rwkv_src.rwkv_model import RWKV_RNN, RWKV_Config
import types
import os, sys
import torch
import numpy as np
import argparse
from pathlib import Path
from rwkv_src.rwkv_model_utils import get_dummy_state_kvcache, sample_logits

parser = argparse.ArgumentParser(description='test torch model')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
parser.add_argument('calib_dataset', type=Path, help='Path to calib text file')
parser.add_argument('--output', type=Path, default='./calib_dataset.npy', help='Path to output file')
parser.add_argument('--block_size', type=int, default=128, help='Block size')
parser.add_argument('--num_samples', type=int, default=1, help='Number of samples')
args = parser.parse_args()

model_args = RWKV_Config()
model_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_args.dtype = torch.float32
model_args.convert_fc_to_conv = False
model_args.rescale_layer = 0
model_args.wkv_customop = False
model_args.output_last = False

model_args.model_name = str(args.model)

tokenizer = RWKV_TOKENIZER("./assets/rwkv_vocab_v20230424.txt")

model = RWKV_RNN(model_args)
device = model.device

with open(args.calib_dataset, 'r') as f:
    text = f.read()
    token_ids = tokenizer.encode(text)

data = {f"{i}": [] for i in range(model.model_info.n_layer * 3 + 1)}
num_samples = min(args.num_samples, len(token_ids) // args.block_size)

for i in range(num_samples):
    states = get_dummy_state_kvcache(1, model.model_info, device)
    for j in range(i * args.block_size, (i + 1) * args.block_size):
        token_input = torch.LongTensor([token_ids[j]]).to(device=device)
        for k in range(len(states)):
            data[f"{k}"].append(states[len(states) - k - 1].squeeze(0).cpu().float().detach().numpy().astype(np.float32))
        data[f"{len(states)}"].append(token_input.squeeze(0).cpu().detach().numpy().astype(np.int32))

        _, states = model(token_input, *states)

for i in range(model.model_info.n_layer * 3 + 1):
    data[f"{i}"] = np.stack(data[f"{i}"], axis=0)

np.save(args.output, data)
