# Report

## Environment

Framework: [Pytorch](https://pytorch.org/)  
Model Pakage: [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models/)

## Introdunction

The homework is to classify 200 species of birds. This problem is called fine-grained image classification.
It's a hard problem in Computer Vision. The data provided from TA is similar as public dataset - [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), but it's not the same. In this homework, it need to use the SOTA model to achieve the high accuracy.

## Data

This project data is from Codalab Competition on class - [2021 VRDL HW1](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07). Include 6,033 bird images belonging to 200 bird species, e.g., tree sparrow or mockingbird, for training images is 3,000 and testing image is 3,033. More important is **external data is NOT allowed to train your model!**

## Methodology

### Data preprocessing

* **Split Dataset**
  * I split the 3,000 training images into training and validation. Training part is 2,400 images, and validation part is 600 images. The model weight is selected by the accuracy of validation.

* **Resize Image**
  * You can select different models, and each of them have different target size. When loading the image, it'll resize the image first. I use `transforms.Resize` to resize the image.
    |      model       | target size |
    | :--------------: | :---------: |
    |     ResNet50     |     224     |
    |   Inception V3   |     299     |
    | Swin Transformer |     224     |
    | Swin Transformer |     384     |

* **Data Augmentation**
  * Augmentation on Training part
    * `transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`
  * Augmentation on Validation part
    * `transforms.CenterCrop(target_size)`
    * `transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`

### Transfer Learning

### Ensemble - Bagging

### Model Architecture

* **ResNet50**
  * [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

* **Inception V3**
  * [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)

* **Swin Transformer**
  * [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)

### Hyperparameters

* **Loss Function**
  * CrossEntropyLoss

* **Optimizer**
  * Adam with learning rate=1e-3 and weight decay=5e-4

* **Epochs**
  * 10 epoch

## Summary

  Something not finish yet.
