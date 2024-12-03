import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import SimpleCNN
import matplotlib.pyplot as plt
import os
import json
import torch.nn as nn
from torchsummary import summary
import random
#import torchvision.transforms.functional as F
import torch.nn.functional as F

# Custom Cutout transformation
class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        w, h = img.size
        mask = F.to_tensor(img).clone()
        for _ in range(self.n_holes):
            y = random.randint(0, h)
            x = random.randint(0, w)

            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)

            mask[:, y1:y2, x1:x2] = 0  # Set the mask to zero (cutout)
        return F.to_pil_image(mask)

torch.manual_seed(1)
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
# Load MNIST dataset with augmentations for training


transform_train = transforms.Compose([
                        
                        #transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
                        #transforms.RandomRotation(5),  # Rotate images by -15 to +15 degrees
                        #transforms.Resize((28, 28)),
                        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # Apply Gaussian Blur
                        transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
                        #transforms.RandomCrop(size=(28, 28), padding=4),  # Random crop with padding
                        #Cutout(n_holes=1, length=16),   # Apply Cutout with 1 hole of length 16
                        #transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2),
                        transforms.ToTensor(), 
                        transforms.Normalize((0.1307,), (0.3081,))
])

# Test dataset without augmentations
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


full_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=None)

#full_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)


# Step 3: Split the dataset into 50k for training and 10k for testing
train_size = 50000
test_size = 10000
trainset, testset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

# Step 4: Create data loaders
batch_size = 128

# Step 4: Apply specific transforms to the train and test datasets
# Wrap the datasets with the appropriate transform
trainset.dataset.transform = transform_train
testset.dataset.transform = transform_test

# Check the sizes of the splits
print (f"Dataset sizes: Full dataset - {len(full_dataset)}, Train dataset - {len(trainset)}, Test dataset - {len(testset)}")
#print(f"Train dataset size: {len(trainset)}")
#print(f"Test dataset size: {len(testset)}")


# Load MNIST training and test datasets
#trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset , batch_size=batch_size, shuffle=True)


testset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_test)

#testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = SimpleCNN().to(device)

# Function to print model summary with layer details
def print_model_summary(model, input_size):
    model.to(device)  # Ensure the model is on the correct device
    summary(model, input_size)

print_model_summary(model, (1, 28, 28))  # MNIST images are 1 channel, 28x28 pixels
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

#optimizer = optim.Adam(model.parameters(), lr=0.1)

from tqdm import tqdm
def train(model, device, train_loader, optimizer, epoch):
    model.train() #Set model to train mode
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() #Preventing gradient accumulation
        output = model(data)
        #loss = criterion(output, target) #Negative log likelihood loss
        loss = F.nll_loss(output, target)
        loss.backward() #Backpropagation. Weight calculation.
        optimizer.step() #Update parameter values
        pbar.set_description(desc= f'epoch={epoch} loss={loss.item()} batch_id={batch_idx}')


def test(model, device, test_loader):
    model.eval() #Set model to test
    test_loss = 0
    correct = 0
    incorrect_samples = []  # To store incorrect predictions

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += criterion(output, target).item()  # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Store incorrect predictions
            for i in range(data.size(0)):
                if pred[i].item() != target[i].item():
                    incorrect_samples.append((data[i].cpu(), target[i].item(), pred[i].item()))

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    # Display some incorrect predictions
    if incorrect_samples and False:
        num_display = min(10, len(incorrect_samples))  # Display up to 10 incorrect predictions
        fig, axes = plt.subplots(1, num_display, figsize=(15, 3))
        
        for i, (image, true_label, pred_label) in enumerate(incorrect_samples[:num_display]):
            image = image.squeeze(0)  # Remove channel dimension
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f' T: {true_label}, P: {pred_label} ')
            axes[i].axis('off')
        
        plt.show()
    
    return accuracy

def train_model(num_epochs=20):

    show_images = True
    if show_images:
        all_images, all_labels = next(iter(torch.utils.data.DataLoader(trainset, batch_size=len(trainset))))
        indices = random.sample(range(len(all_images)), 5)

        # Step 4: Display the selected images
        plt.figure(figsize=(10, 5))
        for i, idx in enumerate(indices):
            image, label = all_images[idx], all_labels[idx]
            plt.subplot(1, 5, i + 1)
            plt.imshow(image.squeeze(), cmap="gray")
            plt.title(f"Label: {label.item()}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()  # Show the plot without blocking code execution
        #plt.show(block=False)  # Show the plot without blocking code execution
        #plt.pause(2)  # Pause for 2 seconds
        #plt.close()  # Close the plot window


    for epoch in range(0, num_epochs): #Run for 18 epochs
        train(model, device, trainloader, optimizer, epoch)
    
    return model

def train_and_test_model(num_epochs=20):
    for epoch in range(0, num_epochs): #Run for 18 epochs
        train(model, device, trainloader, optimizer, epoch)
        #print("Evaluating model: Training accuracy")
        #evaluate_model(testdata=0)
        print("Evaluating model: Test accuracy")
        test(model, device, testloader)

def evaluate_model(testdata=1):
    print(f"Evaluating model: {testdata}", end="")
    if testdata == 1:
        return test(model, device, testloader)
    else:
        return test(model, device, trainloader)

def evaluate_model_old():
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')
# Training function
def train_model_old(num_epochs=5):
    loss_list = []
    accuracy_list = []  # List to store accuracy values
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = 100 * correct / total  # Calculate accuracy
        loss_list.append(epoch_loss)
        accuracy_list.append(epoch_accuracy)  # Append accuracy to the list
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
        
        # Save loss and accuracy to JSON for logging
        with open('loss_log.json', 'w') as f:
            json.dump({'loss': loss_list, 'accuracy': accuracy_list}, f)

    # Evaluate on test set after training
    evaluate_model()

# Function to evaluate the model on the test set


# Start training
if __name__ == "__main__":
    #train_model_old()
    train_and_test_model() 
