from rwkv_src.rwkv_tokenizer import RWKV_TOKENIZER
import types
import os, sys
import torch
import numpy as np
import argparse
from pathlib import Path
from rwkv_src.rwkv_model_utils import get_dummy_state_kvcache, sample_logits

from libnoe import *
import NOE_Engine.NOE_Engine as NOE_Engine

parser = argparse.ArgumentParser(description='test cix model on npu')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
args = parser.parse_args()

model_args.model_name = str(args.model)

tokenizer = RWKV_TOKENIZER("./rwkv_src/rwkv_vocab_v20230424.txt")

engine = NOE_Engine.EngineInfer(str(args.model))

states = get_dummy_state_kvcache(1, model.model_info, device)
token_input = np.array([0], dtype=np.int32)
states = [s.numpy().astype(np.float16) for s in states]
outputs = engine.forward()
print(outputs)

# prompt = 'The Eiffel Tower is in the city of'
# # prompt = """User: 请为我写一首诗。\n\nAssistant:"""
# prompt_ids = tokenizer.encode(prompt)
# print(prompt, end='', flush=True)

# for token in prompt_ids:
#     # logits, states = model(torch.LongTensor([token]).to(device=device), *states)
# logits = logits[:, -1, :]

# for i in range(300):
#     # next_token = sample_logits(logits)
#     next_token = np.argmax(logits.cpu().float().detach().numpy())
#     try:
#         print(tokenizer.decode([next_token]), end='', flush=True)
#     except:
#         print(logits)
#         quit()
#     logits, states = model(torch.LongTensor([next_token]).to(device=device), *states)

# print("\n")

engine.clean()