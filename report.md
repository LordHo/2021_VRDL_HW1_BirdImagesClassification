# Report

## Environment

Framework: [Pytorch](https://pytorch.org/)  
Model Pakage: [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models/)

## Introdunction

The homework is to classify 200 species of birds. This problem is called fine-grained image classification. It's a hard problem in Computer Vision. The data provided from TA is similar as public dataset - [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), but it's not the same. In this homework, it need to use the state-of-the-art models to achieve the high accuracy.

## Data

This project data is from Codalab Competition on class - [2021 VRDL HW1](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07). Include 6,033 bird images belonging to 200 bird species, e.g., tree sparrow or mockingbird, for training images is 3,000 and testing image is 3,033. More important is **external data is NOT allowed to train your model!**

## Methodology

### Data preprocessing

* **Split Dataset**
  * I split the 3,000 training images into training and validation. Training part is 2,400 images, and validation part is 600 images.  
    The model weight is selected by the accuracy of validation.

* **Resize Image**
  * You can select different models, and each of them have different target size. When loading the image, it'll resize the image first.  
    I use `transforms.Resize` to resize the image.
    |      model       | target size |
    | :--------------: | :---------: |
    |     ResNet50     |     224     |
    |   Inception V3   |     299     |
    | Swin Transformer | 224 or 384  |

* **Data Augmentation**
  * Augmentation on Training part
    * `transforms.RandomRotation(degrees=(-45, 45))`
    * `transforms.RandomHorizontalFlip(p=0.5)`
    * `transforms.RandomVerticalFlip(p=0.5)`
    * `transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`
  * Augmentation on Validation part
    * `transforms.CenterCrop(target_size)`
    * `transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`

### Transfer Learning

The data contains images similar to those in ImageNet, we use model that has been pretrained on ImageNet. I replace the output of final fully connected layer in model to 200 classes and fine-tune the model to fit our data. In this homework, various state-of-the-art models are tested to get higher performace, such as ResNet-50 [[1]](https://arxiv.org/abs/1512.03385), Inception V3 [[2]](https://arxiv.org/abs/1409.4842), Swin Transformer [[3]](https://arxiv.org/pdf/2103.14030.pdf), etc.

### Model Architecture

* **ResNet50**
  * Deep Residual Learning for Image Recognition [[1]](https://arxiv.org/abs/1512.03385)

* **Inception V3**
  * Going Deeper with Convolutions [[2]](https://arxiv.org/abs/1409.4842)

* **Swin Transformer**
  * Swin Transformer: Hierarchical Vision Transformer using Shifted Windows [[3]](https://arxiv.org/pdf/2103.14030.pdf)

### Hyperparameters

* **Loss Function**
  * CrossEntropyLoss

* **Optimizer**
  * I try Adam and SGD in different learning rate and weight decay. The result is in following table. Momentum of SGD fix at 0.9.
    | model                | Optimizer | Learning Rate | Weight Decay | Codalab Accuracy | Training Accuracy | Validation Accuracy |
    | -------------------- | --------- | ------------- | ------------ | ---------------- | ----------------- | ------------------- |
    | swin transformer 384 | Adam      | 5.0e-04       | 5.0e-05      | 0.728322         | 0.9967            | 0.8983              |
    |                      | SGD       | 1.0e-02       | 1.0e-03      |                  | 0.9950            | 0.8983              |
    |                      | SGD       | 8.0e-03       | 8.0e-04      |                  | 0.9967            | 0.8900              |
    |                      | SGD       | 5.0e-03       | 5.0e-04      |                  | 0.9950            | 0.8967              |
    |                      | SGD       | 3.0e-03       | 3.0e-04      |                  | 0.9908            | 0.8933              |
    |                      | SGD       | 1.0e-03       | 1.0e-04      |                  | 0.9646            | 0.8767              |
    |                      | SGD       | 8.0e-04       | 8.0e-05      |                  | 0.9621            | 0.8600              |
    |                      | SGD       | 5.0e-04       | 5.0e-05      |                  | 0.9367            | 0.8417              |
    |                      | SGD       | 3.0e-04       | 3.0e-05      |                  | 0.8896            | 0.8017              |
    |                      | SGD       | 1.0e-04       | 1.0e-05      |                  | 0.6204            | 0.6150              |

* **Epochs**
  * 10 epoch

## Results

Our model achieves the following performance on :

**[2021 VRDL HW1](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07)**

|    Model name    | Pre-trained | image size | Top 1 Accuracy |
| :--------------: | :---------: | :--------: | :------------: |
|     ResNet50     |  ImageNet   |    224     |    0.368942    |
|   Inception V3   |  ImageNet   |    299     |    0.285526    |
| Swin transformer |  ImageNet   |    224     |    0.712826    |
| Swin transformer |  ImageNet   |    384     |    0.737554    |

## Summary

I select RestNet and Inception that are familiar to me in the begining. However, performance of both model are frustrating. Then, I remember that teacher introducted Vision Transformer and Swin Transformer in class. Swin Transformer performs better than Vision Transformer on ImageNet dataset, so I choose Swin Transformer for my model. After using Swin Transformer, the top 1 accuracy is double as performance by using RestNet. This result excites me a lot. Then, I continue thinking the effect of the Optimizer, so I do a liitle experiment about it.  
The appropriate learning rate is a range, in that range, the performance is amost the same. However, if we drop the learning rate more, the performance will also drop. I learn that the model can get a enormous progress in performance, so choosing a appropriate model is necessary.

## References

[1] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)  
[2] [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)  
[3] [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)  
[4] [CarClassifier](https://github.com/Yunyung/CarClassifier)  
[5] [FINETUNING TORCHVISION MODELS](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
