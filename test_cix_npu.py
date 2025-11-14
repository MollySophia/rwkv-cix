from rwkv_src.rwkv_tokenizer import RWKV_TOKENIZER
import types
import os, sys
import torch
import numpy as np
import argparse
from pathlib import Path
from rwkv_src.rwkv_model_utils import get_dummy_state_kvcache, sample_logits
import time

# import NOE_Engine.NOE_Engine as NOE_Engine
from utils.NOE_Engine import EngineInfer

parser = argparse.ArgumentParser(description='test cix model on npu')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
args = parser.parse_args()

tokenizer = RWKV_TOKENIZER("./rwkv_src/rwkv_vocab_v20230424.txt")

engine = EngineInfer(str(args.model))
num_layers = 12

inputs = []
for _ in range(num_layers):
    inputs.append(np.zeros((1, 1, 768), dtype=np.float16))
    inputs.append(np.zeros((12, 64, 64), dtype=np.float16))
    inputs.append(np.zeros((1, 1, 768), dtype=np.float16))
inputs.append(np.array([0], dtype=np.int32))

# t0 = time.perf_counter()
# for _ in range(64):
#     outputs = engine.forward(inputs)

# t1 = time.perf_counter()
# print(f"Decode: {64 / (t1 - t0)} tokens/s")

# print(outputs)

prompt = 'The Eiffel Tower is in the city of'
# prompt = """User: 请为我写一首诗。\n\nAssistant:"""
prompt_ids = tokenizer.encode(prompt)
print(prompt, end='', flush=True)

for token in prompt_ids:
    inputs[-1][0] = token
    outputs = engine.forward(inputs)
    inputs[:-1] = outputs[-1:0:-1]
    logits = outputs[0]

for i in range(300):
    # next_token = sample_logits(logits)
    next_token = np.argmax(logits)
    try:
        print(tokenizer.decode([next_token]), end='', flush=True)
    except:
        print(logits)
        quit()
    inputs[-1][0] = next_token
    outputs = engine.forward(inputs)
    inputs[:-1] = outputs[-1:0:-1]
    logits = outputs[0]

print("\n")

engine.clean()