import torch
import torch.nn as nn
import gradio as gr
from transformers import AutoTokenizer
import logging
import gdown
import os
from dataclasses import dataclass
import math
import torch.nn.functional as F

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GPTConfig:
    block_size: int = 2048  # max sequence length
    vocab_size: int = 49152  # vocab size from the notebook
    n_layer: int = 30  # number of layers
    n_head: int = 9  # number of heads
    n_embd: int = 576  # embedding dimension

# Copy the model architecture from the notebook
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                   .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # Flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Google Drive file ID and checkpoint path
CHECKPOINT_GDRIVE_URL = "https://drive.google.com/file/d/1fTezLF3W91awaGn2zkWz1oIM5BnAjbTj/view?usp=sharing"
CHECKPOINT_PATH = "llm_checkpoint.pt"

def get_file_id_from_url(url):
    """Extract file ID from Google Drive share URL"""
    try:
        if 'drive.google.com/file/d/' in url:
            file_id = url.split('/file/d/')[1].split('/')[0]
        elif 'drive.google.com/open?id=' in url:
            file_id = url.split('id=')[1]
        else:
            raise ValueError("Invalid Google Drive URL format")
        return file_id
    except Exception as e:
        logger.error(f"Error extracting file ID: {str(e)}")
        raise

def download_checkpoint():
    """Download checkpoint from Google Drive if it doesn't exist"""
    try:
        if not os.path.exists(CHECKPOINT_PATH):
            logger.info("Downloading checkpoint from Google Drive...")
            file_id = get_file_id_from_url(CHECKPOINT_GDRIVE_URL)
            download_url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(download_url, CHECKPOINT_PATH, quiet=False)
            logger.info("Checkpoint downloaded successfully")
        else:
            logger.info("Checkpoint file already exists")
    except Exception as e:
        logger.error(f"Error downloading checkpoint: {str(e)}")
        raise

# Add this function to inspect checkpoint keys
def inspect_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    print("Keys in checkpoint:")
    for k in state_dict.keys():
        print(k)

# Update the TextGenerator class
class TextGenerator:
    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        try:
            # Ensure checkpoint is downloaded
            download_checkpoint()
            
            # First inspect the checkpoint structure
            logger.info("Inspecting checkpoint keys...")
            inspect_checkpoint(checkpoint_path)
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Initialize model with the same configuration as training
            self.model = GPT(GPTConfig()).to(self.device)
            logger.info("Model initialized successfully")
            
            # Load checkpoint
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            
            # Handle both cases: direct state dict or wrapped in 'model_state_dict'
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Print expected keys from model
            logger.info("Expected keys in model:")
            for k in self.model.state_dict().keys():
                logger.info(f"  {k}")
            
            # Try to load state dict with strict=False first to see what matches
            self.model.load_state_dict(state_dict, strict=False)
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
            logger.info("Tokenizer initialized successfully")
            
            self.model.eval()
        except Exception as e:
            logger.error(f"Error in initialization: {str(e)}")
            raise

    def generate(self, prompt, max_new_tokens=100, temperature=0.8, top_k=40):
        try:
            logger.info(f"Generating text for prompt: {prompt}")
            logger.info(f"Parameters: max_tokens={max_new_tokens}, temp={temperature}, top_k={top_k}")
            
            # Tokenize input
            encoded = self.tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
            
            # Decode and return
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            logger.info("Text generation completed successfully")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            return f"Error in text generation: {str(e)}"

# Initialize the generator
try:
    logger.info("Initializing TextGenerator")
    generator = TextGenerator()
    logger.info("TextGenerator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize TextGenerator: {str(e)}")
    raise

def generate_text(input_text, num_tokens, temperature=0.8, top_k=40):
    try:
        logger.info(f"Received request - Input: {input_text}, Tokens: {num_tokens}")
        result = generator.generate(
            input_text,
            max_new_tokens=int(num_tokens),
            temperature=temperature,
            top_k=top_k
        )
        logger.info("Text generation completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in generate_text: {str(e)}")
        return f"Error: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Input Text", placeholder="Enter your prompt here..."),
        gr.Number(label="Number of tokens to generate", value=100, minimum=1, maximum=1000),
        gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1),
        gr.Slider(label="Top-k", minimum=1, maximum=100, value=40, step=1)
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Text Generation with GPT",
    description="Enter some text and the model will continue it.",
    examples=[
        ["Once upon a time", 50, 0.8, 40],
        ["The quick brown fox", 30, 0.7, 30],
        ["In a galaxy far far away", 40, 0.9, 50]
    ]
)

# Launch the app
if __name__ == "__main__":
    logger.info("Starting Gradio interface")
    iface.launch() 
