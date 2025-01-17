import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet152
from PIL import Image
import requests
import gdown
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the Params class needed for loading the checkpoint
class Params:
    def __init__(self):
        self.batch_size = 48
        self.name = "resnet_152_sgd1"
        self.workers = 4
        self.lr = 0.002
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.lr_step_size = 30
        self.lr_gamma = 0.1

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

# Download ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
response = requests.get(LABELS_URL)
labels = {str(i): label.strip() for i, label in enumerate(response.text.split("\n"))}

def get_file_id_from_url(url):
    """Extract file ID from Google Drive share URL"""
    try:
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

def download_weights(url, output_path='imagenet.pth'):
    """Download checkpoint from Google Drive if it doesn't exist"""
    try:
        if not os.path.exists(output_path):
            logger.info("Downloading weights from Google Drive...")
            file_id = get_file_id_from_url(url)
            download_url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(download_url, output_path, quiet=False)
            logger.info("Weights downloaded successfully")
        else:
            logger.info("Weight file already exists")
        return output_path
    except Exception as e:
        logger.error(f"Error downloading weights: {str(e)}")
        raise

# Load model and set to evaluation mode
try:
    # Initialize model
    model = resnet152(pretrained=False)
    
    # Download and load checkpoint
    drive_url = 'https://drive.google.com/file/d/1OqkOM8T9D_87VHmnx0BubH2AE32Pcokm/view?usp=sharing'
    weights_path = download_weights(drive_url)
    
    if not os.path.exists(weights_path):
        raise Exception("Weight file not found after download")
        
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
    
    # Load just the model state dict
    model.load_state_dict(checkpoint['model'])
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise e

model.eval()

# Define image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

# Define prediction function
def predict(image):
    if image is None:
        return "Please upload an image"
    
    # Convert to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Preprocess the image
    input_tensor = preprocess_image(image)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    # Get top 3 predictions
    top3_prob, top3_indices = torch.topk(probabilities, 3)
    
    result = ""
    for i in range(3):
        result += f"{labels[str(top3_indices[i].item())]}: {top3_prob[i].item()*100:.2f}%\n"
    
    return result

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Predictions"),
    title="ImageNet Classification",
    description="Upload an image to classify it using a ResNet152 model trained on ImageNet.",
    examples=[["example1.jpg"], ["example2.jpg"]]
)

# Launch the app
if __name__ == "__main__":
    iface.launch() 