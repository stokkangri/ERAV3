import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

import tiktoken
from transformers import AutoTokenizer

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        #enc = tiktoken.get_encoding('gpt2')
        # Load the HuggingFace tokenizer
        enc = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets

        # Debugging tensor shapes
        #print(f"x shape: {x.shape}, y shape: {y.shape}")

        # Advance the position in the tensor
        self.current_position += B*T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# CHANGES IN CURRENT CODE
torch.set_float32_matmul_precision('high')
