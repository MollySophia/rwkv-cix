import numpy as np
import torch
from rwkv_src.rwkv_model import RWKV_RNN, RWKV_Config, make_chunks
from rwkv_src.rwkv_model_utils import get_dummy_state_kvcache

import os, types
import gc

args = RWKV_Config()
args.model_name = '/models/rwkv7-g1a-0.1b-20250728-ctx4096'

args.dtype = torch.float32
args.convert_fc_to_conv = False
args.device = 'cpu'
args.rescale_layer=0

print(f'Loading {args.model_name} ...')
model = RWKV_RNN(args)
model = model.to(args.device)

# input = torch.zeros(1, 1, model.model_info.n_embd, device=args.device, dtype=args.dtype)
in0 = torch.LongTensor([24281]).to(args.device)
state = get_dummy_state_kvcache(1, model.model_info, torch.device(args.device), args.dtype)

input_names = ['input'] + [f'state{i}_in' for i in range(len(state))]
output_names = ['output'] + [f'state{i}_out' for i in range(len(state))]
torch.onnx.export(model, (in0, *(state)), "model.onnx", verbose=False, input_names=input_names, output_names=output_names)
