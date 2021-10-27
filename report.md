###### tags: `Image Classification`

## Environment
Framework: Pytorch

## Introdunction

## Data
This project data is from Codalab Competition on class.
This is the link of Competition - [2021 VRDL HW1](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07).

Include 6,033 bird images belonging to 200 bird species, 
e.g., tree sparrow or mockingbird, for training images is 3,000 and testing image is 3,033.
More important is **external data is NOT allowed to train your model!**

## Methodology

### Data preprocessing
- **Split Dataset**
- **Resize Image**
- **Data Augmentation**

### Transfer Learning

### Ensemble - Bagging

### Model Architecture
- **ResNet50**
    - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Inception V3**
    - [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)

### Hyperparameters
- **Loss Function**
    - CrossEntropyLoss
- **Optimizer** 
    - SGD with learning rate = 1e-3 and momentum = 0.9
- **Epochs** 
    - 50 epoch

## Summary