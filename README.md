# 2021 VRDL HW1: Bird Images Classification

This repository is the HW1 of 2021 Selected Topics in Visual Recognition using Deep Learning in NYCU.

## Environment

Framework: [Pytorch](https://pytorch.org/)  

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

To `train.py` and `eval.py` directory:

```src
cd src
```

## Training

To train the model(s) in the project, run this command:

```train
python train.py
```

> 📋  Please modify `model_name` in `train.py` to the model you wanted.  
> 📋  Model name can be select in `['resnet', 'inception', 'swin_transformer_base_224', 'swin_transformer_large_384']`.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py
```

>📋  Please modify `model_name` in `eval.py` to the model you wanted,  
>📋  and `timestamp` in `eval.py` to the training weights that you want to load.

## Pre-trained Models

You can download pretrained models here:

- [My Best Swin Tranformer model](https://drive.google.com/file/d/1mGi_8fKZ5plJixrnPbxnf_OCT_n439WK/view?usp=sharing) pretrained on ImageNet using image size (3, 384, 384).

## Results

Our model achieves the following performance on :

#### [2021 VRDL HW1](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07)

|    Model name    | Pre-trained | image size | Top 1 Accuracy |
| :--------------: | :---------: | :--------: | :------------: |
|     ResNet50     |  ImageNet   |    224     |    0.368942    |
|   Inception V3   |  ImageNet   |    299     |    0.285526    |
| Swin transformer |  ImageNet   |    224     |    0.712826    |
| Swin transformer |  ImageNet   |    384     |    0.737554    |
