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
args = parser.parse_args()

model_args = RWKV_Config()
model_args.device = 'cpu'
model_args.dtype = torch.float16
model_args.convert_fc_to_conv = False
model_args.rescale_layer = 0
model_args.wkv_customop = False
model_args.output_last = False

model_args.model_name = str(args.model)

tokenizer = RWKV_TOKENIZER("./rwkv_src/rwkv_vocab_v20230424.txt")

model = RWKV_RNN(model_args)
device = model.device

prompt = 'The Eiffel Tower is in the city of'
# prompt = """User: 请为我写一首诗。\n\nAssistant:"""
prompt_ids = tokenizer.encode(prompt)
print(prompt, end='', flush=True)

states = get_dummy_state_kvcache(1, model.model_info, device)
for token in prompt_ids:
    logits, states = model(torch.LongTensor([token]).to(device=device), *states)
logits = logits[:, -1, :]

for i in range(300):
    # next_token = sample_logits(logits)
    next_token = np.argmax(logits.cpu().float().detach().numpy())
    try:
        print(tokenizer.decode([next_token]), end='', flush=True)
    except:
        print(logits)
        quit()
    logits, states = model(torch.LongTensor([next_token]).to(device=device), *states)

print("\n")
