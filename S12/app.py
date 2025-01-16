import torch
import torch.nn.functional as F
from transformer import Config, DecoderOnlyTransformer
import gradio as gr
from pathlib import Path
import logging
import gdown
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_file_id_from_url(url):
    """Extract file ID from Google Drive share URL"""
    try:
        # Handle different URL formats
        if 'drive.google.com/file/d/' in url:
            # Format: https://drive.google.com/file/d/FILEID/view?usp=sharing
            file_id = url.split('/file/d/')[1].split('/')[0]
        elif 'drive.google.com/open?id=' in url:
            # Format: https://drive.google.com/open?id=FILEID
            file_id = url.split('id=')[1]
        else:
            raise ValueError("Invalid Google Drive URL format")
        return file_id
    except Exception as e:
        logger.error(f"Error extracting file ID: {str(e)}")
        raise

# Google Drive file ID and checkpoint path
CHECKPOINT_GDRIVE_URL = "https://drive.google.com/file/d/1iee967ih5XPrtXB_8HMuh7Izn7D3W3b0/view?usp=sharing"  # Paste your full share URL here
CHECKPOINT_PATH = "checkpoint_epoch_1.pt"

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

# Model configuration - make sure this matches your training configuration
config = Config(
    vocab_size=50257,
    max_seq_len=256,
    dim=768,
    num_layers=8,
    num_heads=8,
    dropout=0.1
)

class TextGenerator:
    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        try:
            # Ensure checkpoint is downloaded
            download_checkpoint()
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Initialize model
            self.model = DecoderOnlyTransformer(config).to(self.device)
            logger.info("Model initialized successfully")
            
            # Load checkpoint
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Checkpoint loaded successfully")
            
            self.model.eval()
            self.load_char_mappings()
        except Exception as e:
            logger.error(f"Error in initialization: {str(e)}")
            raise
    
    def load_char_mappings(self):
        try:
            # Load your training data to recreate character mappings
            data_dir = Path("data/")
            files = list(data_dir.glob('*.txt'))
            logger.info(f"Found {len(files)} text files in data directory")
            
            if not files:
                raise ValueError("No text files found in data directory")
            
            # Create character mappings
            all_text = ''
            for file in files:
                with open(file, 'r', encoding='utf-8') as f:
                    all_text += f.read()
            
            chars = sorted(list(set(all_text)))
            logger.info(f"Vocabulary size: {len(chars)}")
            logger.debug(f"Characters in vocabulary: {chars}")
            
            self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
            self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
            
            logger.info("Character mappings created successfully")
        except Exception as e:
            logger.error(f"Error in load_char_mappings: {str(e)}")
            raise
    
    def generate(self, prompt, max_new_tokens=100, temperature=0.8):
        try:
            logger.info(f"Generating text for prompt: {prompt}")
            logger.info(f"Max new tokens: {max_new_tokens}, Temperature: {temperature}")
            
            # Check if all characters in prompt are in vocabulary
            unknown_chars = [ch for ch in prompt if ch not in self.char_to_idx]
            if unknown_chars:
                raise ValueError(f"Unknown characters in prompt: {unknown_chars}")
            
            # Convert prompt to tensor
            context = torch.tensor([self.char_to_idx[ch] for ch in prompt], dtype=torch.long)
            context = context.unsqueeze(0).to(self.device)
            logger.debug(f"Input tensor shape: {context.shape}")
            
            generated = list(prompt)
            
            self.model.eval()
            with torch.no_grad():
                for i in range(max_new_tokens):
                    # Get predictions
                    logits = self.model(context)
                    logits = logits[:, -1, :] / temperature
                    
                    # Sample from the distribution
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Add the predicted token to the sequence
                    next_char = self.idx_to_char[next_token.item()]
                    generated.append(next_char)
                    context = torch.cat([context, next_token], dim=1)
                    
                    if i % 20 == 0:  # Log every 20 tokens
                        logger.debug(f"Generated {i} tokens. Last token: {next_char}")
            
            result = ''.join(generated)
            logger.info(f"Successfully generated text of length {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            return f"Error in text generation: {str(e)}\nCheck the logs for more details."

# Initialize the generator
try:
    logger.info("Initializing TextGenerator")
    generator = TextGenerator()
    logger.info("TextGenerator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize TextGenerator: {str(e)}")
    raise

# Define Gradio interface
def generate_text(input_text, num_tokens):
    try:
        logger.info(f"Received request - Input: {input_text}, Tokens: {num_tokens}")
        num_tokens = int(num_tokens)
        result = generator.generate(input_text, max_new_tokens=num_tokens)
        logger.info("Text generation completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in generate_text: {str(e)}")
        return f"Error: {str(e)}\nPlease check the application logs for more details."

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Input Text", placeholder="Enter your prompt here..."),
        gr.Number(label="Number of tokens to generate", value=100, minimum=1, maximum=1000)
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Text Generation with Transformer",
    description="Enter some text and the model will continue it.",
    examples=[
        ["Once upon a time", 10],
        ["The quick brown fox", 15],
        ["In a galaxy far far away", 20]
    ]
)

# Launch the app
if __name__ == "__main__":
    logger.info("Starting Gradio interface")
    iface.launch() 
