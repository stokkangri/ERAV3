from dataloaderlite import DataLoaderLite
from transformers import AutoTokenizer
import torch

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# SEED
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# STOP
num_return_sequences = 5
max_length = 30

train_loader = DataLoaderLite(B = 16, T = 1024)
# Define the fixed text for prediction
fixed_text = "This is a fixed text used for prediction."
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
encoded_text = tokenizer(fixed_text, return_tensors="pt").to(device)
print (f"Encoded Text {encoded_text}")

# Use
from deepseek_smollm2 import DeepSeekModel, DeepSeekConfig
model = DeepSeekModel(DeepSeekConfig())

# Wrap model in DataParallel
#if torch.cuda.device_count() > 1:
#    print(f"Using {torch.cuda.device_count()} GPUs!")
#    model = torch.nn.DataParallel(model)  # Enable Multi-GPU

model.to(device)
#model = torch.compile(model)

# NEW CODE
import time
optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)

# Training loop
total_steps = 10000
prediction_interval = 500
checkpoint_path = "llm_checkpoint.pt"

for step in range(total_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # NEW CODE ADDED HERE
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y) 
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize() 
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f'step{step} | loss: {loss.item()} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec: .2f}')
    
    # Perform prediction every 500 steps
    if (step + 1) % prediction_interval == 0:
        with torch.no_grad():
            model.eval()
            
            max_new_tokens = 10
            temperature = 1.0
            top_k = 40
            
            prediction_logits = model.generate(encoded_text["input_ids"], max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
            prediction = tokenizer.decode(prediction_logits[0], skip_special_tokens=True)
            print(f"Prediction at step {step+1}: \n : {prediction} \n")
            model.train()

# Save checkpoint after 5000 steps
torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, checkpoint_path)
print(f"Checkpoint saved to {checkpoint_path}")