from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.dataset import Dataset
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from tqdm import tqdm

class_dict = {}
train_class = {}


def load_classes():
    classes_file_path = os.path.join('..', 'data', 'classes.txt')
    classes_file = open(classes_file_path, 'r')
    for line in classes_file.readlines():
        line_ = line.rstrip('\n')
        class_num, class_name = line_.split('.')
        class_dict[class_name] = class_num


def load_tain_class():
    species = os.listdir(os.path.join('..', 'data', 'train'))
    for i, specie in enumerate(species):
        train_class[i+1] = specie


class TestDataset(Dataset):
    def __init__(self, image_order_path, image_folder, input_size):
        self.image_order_path = image_order_path
        self.image_folder = image_folder
        self.images = [name.rstrip('\n') for name in open(
            self.image_order_path, 'r').readlines()]
        self.transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path)

        return self.transforms(image), image_name

    def __len__(self):
        return len(self.images)


def load_model(model_path):
    return torch.load(model_path)


def test_model(model, dataloader, result_path):
    result = open(result_path, 'w')

    model.eval()

    # counts = 0

    for inputs, inputs_name in tqdm(dataloader):
        inputs = inputs.to(device)

        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)

        name = train_class[int(preds[0])+1]
        res = class_dict[name]
        result.write(f'{inputs_name[0]} {res}.{name}\n')

        # counts += 1
        # if counts == 5:
        #     break

    result.close()


if __name__ == '__main__':
    # Load trained model
    model_name = 'resnet_2021-10-28 12-00-17'
    model_path = os.path.join('..', 'model', f'{model_name}.pkl')
    model = load_model(model_path)

    timestamp = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())

    # Create training and validation datasets
    image_dataset = TestDataset(os.path.join(
        '..', 'data', 'testing_img_order.txt'), os.path.join('..', 'data', 'test'), input_size=224)
    # Create training and validation dataloaders
    dataloader = DataLoader(image_dataset, batch_size=1)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model = model.to(device)

    load_classes()
    load_tain_class()

    result_path = os.path.join(
        '..', 'result', f'{model_name}_test_{timestamp}.txt')
    test_model(model, dataloader, result_path)
