import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from model import Net
from dataset import DatasetLoader
import multiprocessing
from func_train_test import train, test
import matplotlib.pyplot as plt
import numpy as np

def show_sample_images(dataloader, classes, num_images=20):
    """
    Show sample images from the training dataset
    """
    # Get a batch of training images
    images, labels = next(iter(dataloader))
    
    # Create figure
    fig = plt.figure(figsize=(20, 10))
    
    # Show images
    for i in range(min(num_images, len(images))):
        ax = fig.add_subplot(4, 5, i + 1)
        
        # Convert tensor to numpy and transpose to correct format
        img = images[i].numpy().transpose(1, 2, 0)
        
        # Denormalize the image
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(f'Class: {classes[labels[i]]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_training_images.png')
    plt.close()
    
    # Print class distribution in this batch
    unique_labels, counts = np.unique(labels.numpy(), return_counts=True)
    print("\nClass distribution in sample batch:")
    for label, count in zip(unique_labels, counts):
        print(f"{classes[label]}: {count} images")

def show_misclassified(images, predictions, targets, classes):
    """
    Show misclassified images in a grid
    """
    fig = plt.figure(figsize=(20, 10))
    for i in range(min(20, len(images))):  # Show up to 20 images
        ax = fig.add_subplot(4, 5, i + 1)
        # Convert tensor to numpy and transpose to correct format
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        # Denormalize the image
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(f'Pred: {classes[predictions[i]]}\nTrue: {classes[targets[i]]}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'misclassified_examples.png')
    plt.close()

def plot_training_curves(train_losses, test_losses, train_accs, test_accs, show=True):
    """
    Plot training curves of loss and accuracy
    Args:
        show: If True, plt.show() will be called (for notebooks)
              If False, plots will only be saved to files
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', marker='o')
    ax1.plot(test_losses, label='Test Loss', marker='o')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy', marker='o')
    ax2.plot(test_accs, label='Test Accuracy', marker='o')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the plots
    plt.savefig('training_curves.png')
    
    if show:
        plt.show()
    else:
        plt.close()
        
def main():
    # Debug flag
    DEBUG = False  # Set to False to disable misclassified images visualization
    
    # Training settings
    EPOCHS = 50
    BATCH_SIZE = 128
    
    # CIFAR10 classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data loaders
    get_data = DatasetLoader(batch_size=BATCH_SIZE)
    train_loader = get_data.train_loader()
    test_loader = get_data.test_loader()
    
    # Show sample training images
    print("\nDisplaying sample training images...")
    show_sample_images(train_loader, classes)
    
    # Model
    model = Net().to(device)
    summary(model, input_size=(3, 32, 32))
    
    # Optimizer and criterion
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch: {epoch}")
        
        # Train
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Test and collect misclassified examples
        test_loss, test_acc, misclassified = test(model, device, test_loader, criterion, debug=DEBUG)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # If in debug mode and we have misclassified examples, show them
        if DEBUG and misclassified and epoch % 5 == 0:  # Show every 5 epochs
            images, predictions, targets = misclassified
            show_misclassified(images, predictions, targets, classes)
        
        scheduler.step()
        # Plot current training curves (update every epoch)
        plot_training_curves(train_losses, test_losses, train_accs, test_accs, show=False)
    
    # Final plots with display
    plot_training_curves(train_losses, test_losses, train_accs, test_accs, show=True)
    
    
    # Return training history for further analysis if needed
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs
    }

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()