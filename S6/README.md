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

epoch=0 loss=0.01640809141099453 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.44it/s] 
Evaluating model: Test accuracy

Test set: Average loss: 0.0457, Accuracy: 9852/10000 (98.52%)

epoch=0 loss=0.05186476185917854 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.53it/s] 
Evaluating model: Test accuracy

Test set: Average loss: 0.0466, Accuracy: 9846/10000 (98.46%)

epoch=1 loss=0.031148681417107582 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.43it/s] 
Evaluating model: Test accuracy

Test set: Average loss: 0.0448, Accuracy: 9847/10000 (98.47%)

epoch=2 loss=0.0026431765872985125 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.83it/s]
Evaluating model: Test accuracy

Test set: Average loss: 0.0251, Accuracy: 9910/10000 (99.10%)

epoch=3 loss=0.010564235039055347 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.69it/s] 
Evaluating model: Test accuracy

Test set: Average loss: 0.0241, Accuracy: 9910/10000 (99.10%)

epoch=4 loss=0.025675460696220398 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.91it/s]  
Evaluating model: Test accuracy

Test set: Average loss: 0.0166, Accuracy: 9939/10000 (99.39%)

epoch=5 loss=0.0029225926846265793 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.84it/s]
Evaluating model: Test accuracy

Test set: Average loss: 0.0149, Accuracy: 9948/10000 (99.48%)

epoch=6 loss=0.05393640697002411 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.13it/s]   
Evaluating model: Test accuracy

Test set: Average loss: 0.0137, Accuracy: 9951/10000 (99.51%)

epoch=7 loss=0.0015578469028696418 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.17it/s] 
Evaluating model: Test accuracy

Test set: Average loss: 0.0159, Accuracy: 9946/10000 (99.46%)

epoch=8 loss=0.0068022808991372585 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.93it/s] 
Evaluating model: Test accuracy

Test set: Average loss: 0.0151, Accuracy: 9952/10000 (99.52%)

epoch=9 loss=0.0005414550541900098 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.81it/s] 
Evaluating model: Test accuracy

Test set: Average loss: 0.0085, Accuracy: 9971/10000 (99.71%)

epoch=10 loss=0.0169388260692358 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.13it/s]    
Evaluating model: Test accuracy

Test set: Average loss: 0.0138, Accuracy: 9944/10000 (99.44%)

epoch=11 loss=0.02039267309010029 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.89it/s]   
Evaluating model: Test accuracy

Test set: Average loss: 0.0067, Accuracy: 9977/10000 (99.77%)

epoch=12 loss=0.0008886688738130033 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.69it/s] 
Evaluating model: Test accuracy

Test set: Average loss: 0.0100, Accuracy: 9961/10000 (99.61%)

epoch=13 loss=0.002259073080495 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.99it/s]     
Evaluating model: Test accuracy

Test set: Average loss: 0.0082, Accuracy: 9971/10000 (99.71%)

epoch=14 loss=0.00964026153087616 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.01it/s]   
Evaluating model: Test accuracy

Test set: Average loss: 0.0091, Accuracy: 9966/10000 (99.66%)

epoch=15 loss=0.00027339570806361735 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.05it/s]
Evaluating model: Test accuracy

Test set: Average loss: 0.0076, Accuracy: 9980/10000 (99.80%)

epoch=16 loss=0.0003780572733376175 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.03it/s] 
Evaluating model: Test accuracy

Test set: Average loss: 0.0063, Accuracy: 9977/10000 (99.77%)

epoch=17 loss=0.011560574173927307 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.06it/s]  
Evaluating model: Test accuracy

Test set: Average loss: 0.0089, Accuracy: 9972/10000 (99.72%)

epoch=18 loss=0.0005756766768172383 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.09it/s] 
Evaluating model: Test accuracy

Test set: Average loss: 0.0072, Accuracy: 9979/10000 (99.79%)

epoch=19 loss=4.747032289742492e-05 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.09it/s] 
Evaluating model: Test accuracy

Test set: Average loss: 0.0114, Accuracy: 9959/10000 (99.59%)
