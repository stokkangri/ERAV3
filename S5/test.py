import torch
from model import SimpleCNN
from app import train_model, train_and_test_model, evaluate_model, test
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

# Function to count the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to evaluate the model on the test set
def evaluate_model_old(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100. * correct / total
    return accuracy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST test dataset
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# Initialize model
model = train_model(1)

# Count parameters
num_parameters = count_parameters(model)
print(f'Number of parameters in SimpleCNN: {num_parameters}')

# Train the model for 1 epoch to check accuracy
# You can include the training code here or call the training function from app.py if needed
# For demonstration, we will assume the model has been trained already

# Evaluate the model
accuracy = evaluate_model()
print(f'Accuracy after training: {accuracy:.2f}%')

# Assertions
assert num_parameters < 25000, "The number of parameters exceeds 25,000."
assert accuracy > 95, "The accuracy is less than 95%."
print("All tests passed successfully!") 