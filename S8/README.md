<h3><b> Session 8 </h3>

<h3><i><b>Advanced Neural Network Architectures</b></i></h2>


---


**Target:**
1. Write a new network that works on CIFAR10 Dataset.
2. has the architecture to C1C2C3C40 (No MaxPooling, but convolutions, where the last one has a stride of 2 instead) (NO restriction on using 1x1) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
3. total RF must be more than 44
4. one of the layers must use Depthwise Separable Convolution
5. one of the layers must use Dilated Convolution
6. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
7. use albumentation library and apply:
    - horizontal flip
    - shiftScaleRotate
    - coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
8. achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.
9. make sure you're following code-modularity (else 0 for full assignment) 

**Model details**
1. Model params - 179,904
2. Optimizer - SGD
3. Variable LR - StepLR
4. Epochs - 20
5. Batch Size - 128
6. Use of Depth-wise separable convolution (groups followed by 1x1 conv)
7. Use of dilated kernel (dilation=2)
8. Use of GAP
9. Use of FC layer.

**Results**
1. 85% accuracy with <200K parameters  
    - Train Accuracy - 85.41%
    - Test Accuracy - 85.48%
    - Accuracy/Loss plot - ![kkdm](./training_history.png)

Torch Summary:
****----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
       BatchNorm2d-2           [-1, 32, 32, 32]              64
              ReLU-3           [-1, 32, 32, 32]               0
            Conv2d-4           [-1, 64, 16, 16]           2,048
       BatchNorm2d-5           [-1, 64, 16, 16]             128
            Conv2d-6           [-1, 64, 32, 32]           2,048
       BatchNorm2d-7           [-1, 64, 32, 32]             128
              ReLU-8           [-1, 64, 32, 32]               0
            Conv2d-9           [-1, 64, 16, 16]             576
      BatchNorm2d-10           [-1, 64, 16, 16]             128
             ReLU-11           [-1, 64, 16, 16]               0
           Conv2d-12           [-1, 64, 16, 16]          36,864
             ReLU-13           [-1, 64, 16, 16]               0
      BatchNorm2d-14           [-1, 64, 16, 16]             128
           Conv2d-15          [-1, 128, 16, 16]           8,192
      BatchNorm2d-16          [-1, 128, 16, 16]             256
             ReLU-17          [-1, 128, 16, 16]               0
           Conv2d-18           [-1, 64, 16, 16]           8,192
      BatchNorm2d-19           [-1, 64, 16, 16]             128
             ReLU-20           [-1, 64, 16, 16]               0
          Dropout-21           [-1, 64, 16, 16]               0
           Conv2d-22             [-1, 96, 8, 8]           6,144
      BatchNorm2d-23             [-1, 96, 8, 8]             192
           Conv2d-24           [-1, 64, 16, 16]          36,864
             ReLU-25           [-1, 64, 16, 16]               0
      BatchNorm2d-26           [-1, 64, 16, 16]             128
           Conv2d-27           [-1, 96, 16, 16]           6,144
      BatchNorm2d-28           [-1, 96, 16, 16]             192
             ReLU-29           [-1, 96, 16, 16]               0
           Conv2d-30             [-1, 96, 8, 8]             864
      BatchNorm2d-31             [-1, 96, 8, 8]             192
             ReLU-32             [-1, 96, 8, 8]               0
           Conv2d-33            [-1, 128, 8, 8]          12,288
      BatchNorm2d-34            [-1, 128, 8, 8]             256
             ReLU-35            [-1, 128, 8, 8]               0
           Conv2d-36             [-1, 96, 8, 8]          12,288
      BatchNorm2d-37             [-1, 96, 8, 8]             192
             ReLU-38             [-1, 96, 8, 8]               0
          Dropout-39             [-1, 96, 8, 8]               0
           Conv2d-40            [-1, 128, 4, 4]          12,288
      BatchNorm2d-41            [-1, 128, 4, 4]             256
           Conv2d-42            [-1, 128, 8, 8]          12,288
      BatchNorm2d-43            [-1, 128, 8, 8]             256
             ReLU-44            [-1, 128, 8, 8]               0
           Conv2d-45            [-1, 128, 4, 4]           1,152
      BatchNorm2d-46            [-1, 128, 4, 4]             256
             ReLU-47            [-1, 128, 4, 4]               0
           Conv2d-48            [-1, 128, 4, 4]          16,384
      BatchNorm2d-49            [-1, 128, 4, 4]             256
             ReLU-50            [-1, 128, 4, 4]               0
          Dropout-51            [-1, 128, 4, 4]               0
AdaptiveAvgPool2d-52            [-1, 128, 1, 1]               0
           Linear-53                   [-1, 10]           1,280
================================================================
Total params: 179,904
Trainable params: 179,904
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.38
Params size (MB): 0.69
Estimated Total Size (MB): 7.07
----------------------------------------------------------------****

Training Epochs:
Epoch: 19
Epoch 19 Train: Loss=0.4537 Acc=84.41%: 100%|█████████████████████████████████████████████████████████████████████████| 391/391 [00:04<00:00, 93.98it/s]

Test set: Average loss: 0.4445, Accuracy: 84.70%


Epoch: 20
Epoch 20 Train: Loss=0.4427 Acc=84.67%: 100%|█████████████████████████████████████████████████████████████████████████| 391/391 [00:04<00:00, 95.36it/s]

Test set: Average loss: 0.4445, Accuracy: 84.88%


Epoch: 21
Epoch 21 Train: Loss=0.4366 Acc=84.90%: 100%|█████████████████████████████████████████████████████████████████████████| 391/391 [00:04<00:00, 87.76it/s]

Test set: Average loss: 0.4325, Accuracy: 85.13%


Epoch: 22
Epoch 22 Train: Loss=0.4261 Acc=85.28%: 100%|█████████████████████████████████████████████████████████████████████████| 391/391 [00:04<00:00, 85.78it/s]

Test set: Average loss: 0.4372, Accuracy: 84.94%


Epoch: 23
Epoch 23 Train: Loss=0.4198 Acc=85.25%: 100%|█████████████████████████████████████████████████████████████████████████| 391/391 [00:04<00:00, 94.32it/s]

Test set: Average loss: 0.4393, Accuracy: 85.01%


Epoch: 24
Epoch 24 Train: Loss=0.4176 Acc=85.50%: 100%|█████████████████████████████████████████████████████████████████████████| 391/391 [00:04<00:00, 89.96it/s]

Test set: Average loss: 0.4434, Accuracy: 84.99%
---
