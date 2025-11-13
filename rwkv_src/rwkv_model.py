########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
import torch.utils.cpp_extension
# from rwkv_src.rwkv_v6_modules import Rwkv6SelfAttention, Rwkv6FeedForward
from rwkv_src.rwkv_v7_modules import Rwkv7SelfAttention, Rwkv7FeedForward
# from rwkv_src.rwkv_v5_modules import Rwkv5SelfAttention, Rwkv5FeedForward
# from rwkv_src.rwkv_v7_modules_conv import Rwkv7SelfAttention, Rwkv7FeedForward
# from aimet_torch.v2.nn.modules.custom import Permute, Concat, Reshape
from rwkv_src.rwkv_model_utils import check_rwkv_info

class RWKV_Config:
    model_name: str = ""
    device: str = "cpu"
    rescale_layer: int = 0
    fuse_pre_ln: bool = True
    convert_fc_to_conv: bool = True
    dtype: torch.dtype = torch.float16
    output_last: bool = False
    wkv_customop: bool = False

class RWKV_Block(nn.Module):
    def __init__(self, state_dict, n_embd, head_size, n_ffn, layer_id, layer_begin, rescale_layer=0, version=6.0, custom_wkv=False, layer_total=0, output_last=False):
        super().__init__()
        self.version = version
        self.layer_id = layer_id
        self.layer_offset = layer_id - layer_begin
        self.layer_total = layer_total
        if self.version == 7:
            self.att = Rwkv7SelfAttention(state_dict, n_embd, head_size, layer_id=layer_id, custom_wkv=custom_wkv)
            self.ffn = Rwkv7FeedForward(state_dict, n_embd, n_ffn, layer_id=layer_id, layer_total=layer_total, output_last=output_last)
        # elif self.version == 6:
        #     self.att = Rwkv6SelfAttention(state_dict, n_embd, head_size, layer_id=layer_id, rescale_layer=rescale_layer, custom_wkv=custom_wkv)
        #     self.ffn = Rwkv6FeedForward(state_dict, n_embd, n_ffn, layer_id=layer_id, rescale_layer=rescale_layer)
        # else:
        #     self.att = Rwkv5SelfAttention(state_dict, n_embd, head_size, version=version, layer_id=layer_id, rescale_layer=rescale_layer)
        #     self.ffn = Rwkv5FeedForward(state_dict, n_embd, n_ffn, layer_id=layer_id, rescale_layer=rescale_layer)
        else:
            assert False, "TODO"

    def forward(self, x, state, v_first=None):
        if self.version == 7:
            x, state[3*self.layer_offset], state[3*self.layer_offset+1], v_first = self.att(x, state[3*self.layer_offset], state[3*self.layer_offset+1], v_first)
            x, state[3*self.layer_offset+2] = self.ffn(x, state[3*self.layer_offset+2])
            return x, state, v_first
        else:
            x, state[3*self.layer_offset], state[3*self.layer_offset+1] = self.att(x, state[3*self.layer_offset], state[3*self.layer_offset+1])
            x, state[3*self.layer_offset+2] = self.ffn(x, state[3*self.layer_offset+2])
            return x, state

class RWKV_RNN(torch.nn.Module):
    def __init__(self, config: RWKV_Config, chunks=1, chunk_idx=0):
        super().__init__()
        self.config = config
        self.eval()

        if '.pth' in config.model_name:
            config.model_name = config.model_name.replace('.pth', '')
        w = torch.load(config.model_name + '.pth', map_location='cpu')
        self.model_info = types.SimpleNamespace()
        self.model_info.n_embd = w['emb.weight'].shape[1]
        self.model_info.vocab_size = w['emb.weight'].shape[0]
        self.model_info.n_att = w['blocks.0.att.key.weight'].shape[0]
        self.model_info.n_ffn = w['blocks.0.ffn.key.weight'].shape[0]
        self.model_info.version, self.model_info.n_layer, self.model_info.n_head = check_rwkv_info(w)
        self.model_info.head_size = self.model_info.n_embd // self.model_info.n_head

        if self.model_info.version == 7:
            self.config.rescale_layer = 0

        if chunk_idx == 0:
            print("Model version:", self.model_info.version)
            print("n_layer:", self.model_info.n_layer)
            print("n_embd:", self.model_info.n_embd)
            print("vocab_size:", self.model_info.vocab_size)
            print("n_att:", self.model_info.n_att)
            print("n_ffn:", self.model_info.n_ffn)

        layers_per_chunk = self.model_info.n_layer // chunks
        self.model_info.layer_begin = chunk_idx * layers_per_chunk
        self.model_info.layer_end = min(self.model_info.n_layer, (chunk_idx + 1) * layers_per_chunk)
        self.model_info.chunk_idx = chunk_idx
        self.model_info.num_chunks = chunks
        print(f"Chunk {chunk_idx}: layers {self.model_info.layer_begin} to {self.model_info.layer_end}")

        if config.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")

        self.device = torch.device(config.device)

        if chunk_idx == 0:
            emb_weight = w['emb.weight']
            if config.fuse_pre_ln:
                emb_weight = F.layer_norm(emb_weight.float(), emb_weight.size()[-1:], weight=w['blocks.0.ln0.weight'].flatten().float(), bias=w['blocks.0.ln0.bias'].flatten().float())
            self.embedding = torch.nn.Embedding.from_pretrained(emb_weight)

            self.pre_ln = nn.LayerNorm(self.model_info.n_embd, eps=1e-5)
            self.pre_ln.weight = nn.Parameter(w['blocks.0.ln0.weight'])
            self.pre_ln.bias = nn.Parameter(w['blocks.0.ln0.bias'])

        self.blocks = nn.ModuleList([RWKV_Block(w, self.model_info.n_embd, self.model_info.head_size, self.model_info.n_ffn, \
            layer_id=i,layer_begin=self.model_info.layer_begin, rescale_layer=self.config.rescale_layer, version=self.model_info.version, \
            custom_wkv=self.config.wkv_customop, layer_total=self.model_info.n_layer, \
            output_last=self.config.output_last) for i in range(self.model_info.layer_begin, self.model_info.layer_end)])

        self.ln_out = nn.LayerNorm(self.model_info.n_embd, eps=1e-5)
        self.ln_out.weight = nn.Parameter(w['ln_out.weight'])
        self.ln_out.bias = nn.Parameter(w['ln_out.bias'])

        if self.config.convert_fc_to_conv:
            # if 'head.bias' in w.keys():
            #     self.head = nn.Conv2d(self.model_info.n_embd, self.model_info.vocab_size, 1, bias=False)
            #     self.head.weight = nn.Parameter(w['head.weight'].view(self.model_info.vocab_size, self.model_info.n_embd, 1, 1))
            #     self.head.bias = nn.Parameter(w['head.bias'].reshape(-1))
            # else:
            #     self.head = nn.Conv2d(self.model_info.n_embd, self.model_info.vocab_size, 1, bias=False)
            #     self.head.weight = nn.Parameter(w['head.weight'].view(self.model_info.vocab_size, self.model_info.n_embd, 1, 1))

            # self.head_pre_reshape = Reshape()
            # self.head_post_reshape = Reshape()
            # self.head_pre_permute = Permute()
            # self.head_post_permute = Permute()
            assert False, "TODO"
        else:
            self.head = nn.Linear(self.model_info.n_embd, self.model_info.vocab_size, bias=True if 'head.bias' in w.keys() else False)
            self.head.weight = nn.Parameter(w['head.weight'])
            if 'head.bias' in w.keys():
                self.head.bias = nn.Parameter(w['head.bias'])

        self.to(self.config.dtype)

        self.to(self.device)

    def forward(self, in0: torch.Tensor, *state, v_first = None):
        state = list(state)
        with torch.no_grad():
            if self.model_info.chunk_idx == 0 and (in0.dtype == torch.int64 or in0.dtype == torch.int32):
                x = self.embedding(in0)
            else:
                x = in0

            if self.model_info.chunk_idx == 0 and not self.config.fuse_pre_ln:
                x = self.pre_ln(x)

            if len(x.shape) == 3:
                batch_size, seq_length, _ = x.size()
            else:
                seq_length, _ = x.size()
                batch_size = 1
                x = x.unsqueeze(0)

            for i in range(self.model_info.layer_begin, self.model_info.layer_end):
                if self.model_info.version == 7:
                    x, state, v_first = self.blocks[i-self.model_info.layer_begin](x, state, v_first)
                else:
                    x, state = self.blocks[i-self.model_info.layer_begin](x, state)
                if self.config.rescale_layer > 0:
                    if (i+1) % self.config.rescale_layer == 0:
                        x = x / 2

            if self.model_info.chunk_idx == self.model_info.num_chunks - 1:
                x = self.ln_out(x)
                if self.config.convert_fc_to_conv:
                    x = self.head_pre_reshape(x, [batch_size, -1, 1, self.model_info.n_embd])
                    x = self.head_pre_permute(x, [0, 3, 2, 1])
                x = self.head(x)
                if self.config.convert_fc_to_conv:
                    x = self.head_post_permute(x, [0, 3, 2, 1])
                    x = self.head_post_reshape(x, [batch_size, -1, self.model_info.vocab_size])
            else:
                x = x.view(batch_size, seq_length, self.model_info.n_embd)
            if self.model_info.version == 7 and self.model_info.chunk_idx == 0 and self.model_info.layer_end < self.model_info.n_layer:
                return x, state, v_first
            else:
                return x, state

def make_chunks(chunks, config):
    return [RWKV_RNN(config, chunks=chunks, chunk_idx=i) for i in range(chunks)]
