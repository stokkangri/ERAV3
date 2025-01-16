import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
from transformer import Config, DecoderOnlyTransformer
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

# Configuration
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 10
DATA_DIR = "data/text_files"

class TextDataset(Dataset):
    def __init__(self, data_dir, seq_length):
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.files = list(self.data_dir.glob('*.txt'))
        
        # Simple tokenization (character-level for this example)
        self.chars = sorted(list(set(''.join(open(f).read() for f in self.files))))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # Load all text
        self.text = ''
        for file in self.files:
            with open(file, 'r') as f:
                self.text += f.read()
        
        # Convert text to indices
        self.data = torch.tensor([self.char_to_idx[ch] for ch in self.text], dtype=torch.long)
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x, y

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_distributed(rank, world_size, config):
    setup(rank, world_size)
    
    # Create model and move it to GPU
    model = DecoderOnlyTransformer(config).to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Create dataset and distributed sampler
    dataset = TextDataset(DATA_DIR, config.max_seq_len)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        pin_memory=True
    )
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        if rank == 0:
            print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc='Training') if rank == 0 else dataloader
        for batch_idx, (x, y) in enumerate(progress_bar):
            x = x.to(rank)
            y = y.to(rank)
            
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if rank == 0:
                total_loss += loss.item()
                if isinstance(progress_bar, tqdm):
                    progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
        
        # Save checkpoint only on rank 0
        if rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f'Average loss: {avg_loss:.4f}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoint_epoch_{epoch+1}.pt')
    
    dist.destroy_process_group()

def main():
    # Model configuration
    config = Config(
        vocab_size=50257,
        max_seq_len=1024,
        dim=768,
        num_layers=12,
        num_heads=12,
        dropout=0.1
    )
    
    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs!")
    
    if world_size > 1:
        mp.spawn(
            train_distributed,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        print("No multiple GPUs found. Running on single GPU or CPU...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DecoderOnlyTransformer(config).to(device)
        # Add single GPU training code here...

if __name__ == "__main__":
    main() 