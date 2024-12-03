import torch
from model import SimpleCNN
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

# Function to count the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to print model summary with layer details
def print_model_summary(model, input_size):
    model.to(device)  # Ensure the model is on the correct device
    summary(model, input_size)

# Function to check if model contains batch normalization
def has_batch_norm(model):
    return any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())

# Function to check if model contains dropout
def has_dropout(model):
    return any(isinstance(module, torch.nn.Dropout) for module in model.modules())

# Function to check if model contains fully connected layer or GAP
def has_fc_or_gap(model):
    has_fc = any(isinstance(module, torch.nn.Linear) for module in model.modules())
    has_gap = any(isinstance(module, torch.nn.AdaptiveAvgPool2d) for module in model.modules())
    return has_fc or has_gap

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = SimpleCNN().to(device)

# Print model summary
print_model_summary(model, (1, 28, 28))

# Count parameters
num_parameters = count_parameters(model)
print(f'Number of parameters in SimpleCNN: {num_parameters}')

# Run architecture tests
def test_model_architecture():
    # Test 1: Parameter count
    assert num_parameters < 25000, f"The number of parameters ({num_parameters}) exceeds 25,000"
    print("✓ Parameter count test passed")

    # Test 2: Batch Normalization
    assert has_batch_norm(model), "Model does not use Batch Normalization"
    print("✓ Batch Normalization test passed")

    # Test 3: Dropout
    assert has_dropout(model), "Model does not use Dropout"
    print("✓ Dropout test passed")

    # Test 4: FC Layer or GAP
    assert has_fc_or_gap(model), "Model does not use either Fully Connected Layer or Global Average Pooling"
    print("✓ FC/GAP layer test passed")

if __name__ == "__main__":
    test_model_architecture()
    print("\nAll architecture tests passed successfully!") 