# ERA-V3 Session 6 Assignment

This repository contains the implementation of a Convolutional Neural Network (CNN) for the MNIST digit classification task, following specific architectural and performance constraints.

## Assignment Requirements

1. **Accuracy Target**: 99.4% validation accuracy
2. **Dataset Split**: 50,000 training images / 10,000 validation images
3. **Parameter Constraint**: Less than 20,000 parameters
4. **Training Duration**: Less than 20 epochs
5. **Architecture Requirements**:
   - Must use Batch Normalization
   - Must use Dropout
   - Optional: Either Fully Connected Layer or Global Average Pooling

## Model Architecture

The implemented CNN architecture consists of:

1. First Convolution Block:
   - Conv2d(1, 16, kernel_size=3)
   - BatchNorm2d(16)
   - Dropout(0.01)

2. Second Convolution Block:
   - Conv2d(16, 32, kernel_size=3)
   - BatchNorm2d(32)
   - MaxPool2d(2)
   - Dropout(0.01)

3. Third Convolution Block:
   - Conv2d(32, 32, kernel_size=3)
   - BatchNorm2d(32)
   - MaxPool2d(2)
   - Dropout(0.00)

4. Dimensionality Reduction:
   - Conv2d(32, 16, kernel_size=1) [1x1 convolution]

5. Output Layer:
   - Fully Connected Layer (16*5*5, 10)
   - LogSoftmax activation

Total Parameters: 18,746

## Data Augmentation

The training data uses the following augmentation techniques:
- Gaussian Blur (kernel_size=5)
- Random Affine Transformations (rotation, translation, scaling)
- Color Jitter (brightness and contrast)
- Normalization (mean=0.1307, std=0.3081)

## Training Details

- Optimizer: SGD with momentum
  - Learning Rate: 0.1
  - Momentum: 0.9
- Loss Function: Negative Log Likelihood Loss
- Batch Size: 128
- Device: GPU (if available) / CPU

## Results

The model achieves:
- Test Accuracy: >99.4%
- Total Parameters: 18,746 (within 20k limit)
- Convergence: Achieved target accuracy within 20 epochs

## Files Structure

- `model.py`: Contains the CNN model architecture
- `test.py`: Contains test cases for model validation
- `app.py`: Training and evaluation scripts
- `README.md`: Project documentation

## Test Cases

The model implementation is verified through test cases that check:
1. Total parameter count (< 20,000)
2. Presence of Batch Normalization layers
3. Presence of Dropout layers
4. Presence of either Fully Connected Layer or Global Average Pooling 

Training run log to show accuracy and loss:

epoch=0 loss=0.04691198840737343 batch_id=390: 100%|██████████| 391/391 [00:24<00:00, 15.82it/s] 
Evaluating model: Test accuracy

Test set: Average loss: 0.0463, Accuracy: 59099/60000 (98.50%)

epoch=1 loss=0.06381608545780182 batch_id=390: 100%|██████████| 391/391 [00:24<00:00, 15.80it/s]  
Evaluating model: Test accuracy

Test set: Average loss: 0.0312, Accuracy: 59419/60000 (99.03%)

epoch=2 loss=0.009553512558341026 batch_id=390: 100%|██████████| 391/391 [00:24<00:00, 15.99it/s] 
Evaluating model: Test accuracy

Test set: Average loss: 0.0290, Accuracy: 59440/60000 (99.07%)

epoch=3 loss=0.03585263714194298 batch_id=390: 100%|██████████| 391/391 [00:24<00:00, 15.81it/s]  
Evaluating model: Test accuracy

Test set: Average loss: 0.0251, Accuracy: 59543/60000 (99.24%)

epoch=4 loss=0.07110023498535156 batch_id=390: 100%|██████████| 391/391 [00:25<00:00, 15.59it/s]  
Evaluating model: Test accuracy

Test set: Average loss: 0.0227, Accuracy: 59565/60000 (99.28%)

epoch=5 loss=0.03431084007024765 batch_id=390: 100%|██████████| 391/391 [00:25<00:00, 15.62it/s]  
Evaluating model: Test accuracy

Test set: Average loss: 0.0174, Accuracy: 59673/60000 (99.45%)

epoch=6 loss=0.01150638610124588 batch_id=390: 100%|██████████| 391/391 [00:24<00:00, 15.97it/s]  
Evaluating model: Test accuracy

Test set: Average loss: 0.0218, Accuracy: 59607/60000 (99.34%)

epoch=7 loss=0.0026571291964501143 batch_id=390: 100%|██████████| 391/391 [00:24<00:00, 16.08it/s] 
Evaluating model: Test accuracy

Test set: Average loss: 0.0129, Accuracy: 59779/60000 (99.63%)

epoch=8 loss=0.006700292229652405 batch_id=390: 100%|██████████| 391/391 [00:24<00:00, 16.05it/s]  
Evaluating model: Test accuracy

Test set: Average loss: 0.0144, Accuracy: 59736/60000 (99.56%)

epoch=9 loss=0.0030945241451263428 batch_id=390: 100%|██████████| 391/391 [00:24<00:00, 15.93it/s] 
Evaluating model: Test accuracy

Test set: Average loss: 0.0189, Accuracy: 59664/60000 (99.44%)

epoch=10 loss=0.10154619067907333 batch_id=390: 100%|██████████| 391/391 [00:24<00:00, 16.02it/s]   
Evaluating model: Test accuracy

Test set: Average loss: 0.0157, Accuracy: 59732/60000 (99.55%)

epoch=11 loss=0.057716239243745804 batch_id=390: 100%|██████████| 391/391 [00:24<00:00, 16.06it/s]  
Evaluating model: Test accuracy

Test set: Average loss: 0.0108, Accuracy: 59845/60000 (99.74%)

epoch=12 loss=0.01438748650252819 batch_id=390: 100%|██████████| 391/391 [00:24<00:00, 16.04it/s]   
Evaluating model: Test accuracy

Test set: Average loss: 0.0190, Accuracy: 59669/60000 (99.45%)

epoch=13 loss=0.009068877436220646 batch_id=390: 100%|██████████| 391/391 [00:24<00:00, 15.78it/s]  
Evaluating model: Test accuracy

Test set: Average loss: 0.0126, Accuracy: 59817/60000 (99.69%)

epoch=14 loss=0.0006239829235710204 batch_id=390: 100%|██████████| 391/391 [00:26<00:00, 14.97it/s] 
Evaluating model: Test accuracy

Test set: Average loss: 0.0130, Accuracy: 59766/60000 (99.61%)

epoch=15 loss=0.05126728489995003 batch_id=390: 100%|██████████| 391/391 [00:24<00:00, 15.92it/s]   
Evaluating model: Test accuracy

Test set: Average loss: 0.0144, Accuracy: 59762/60000 (99.60%)

epoch=16 loss=0.029997151345014572 batch_id=390: 100%|██████████| 391/391 [00:24<00:00, 16.02it/s]  
Evaluating model: Test accuracy

Test set: Average loss: 0.0157, Accuracy: 59724/60000 (99.54%)

epoch=17 loss=0.013998043723404408 batch_id=390: 100%|██████████| 391/391 [00:24<00:00, 16.04it/s]  
Evaluating model: Test accuracy

Test set: Average loss: 0.0119, Accuracy: 59811/60000 (99.69%)

epoch=18 loss=0.055665694177150726 batch_id=390: 100%|██████████| 391/391 [00:24<00:00, 16.04it/s]  
Evaluating model: Test accuracy

Test set: Average loss: 0.0128, Accuracy: 59809/60000 (99.68%)

epoch=19 loss=0.000490246107801795 batch_id=390: 100%|██████████| 391/391 [00:24<00:00, 15.97it/s]  
Evaluating model: Test accuracy

Test set: Average loss: 0.0082, Accuracy: 59902/60000 (99.84%)