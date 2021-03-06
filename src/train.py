from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import timm


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


class Log:
    def __init__(self, result_dir):
        self.log_path = os.path.join(result_dir, 'log.txt')
        self.log = None
        self.row_counts = 0

    def start_loging(self):
        self.log = open(self.log_path, 'w')

    def end_loging(self):
        self.log.close()

    def message_loging(self, message):
        if self.row_counts == 0:
            self.log.write(message)
        else:
            self.log.write('\n' + message)
        self.row_counts += 1


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_optimizer(optimizer_name, lr=None, weight_decay=None):
    # Set the learning rate and weight decay, default is 1e-3 and 1e-4
    lr = 1e-3 if lr is None else lr
    weight_decay = lr*0.1 if weight_decay is None else weight_decay
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(params_to_update, lr=lr,
                               weight_decay=weight_decay)

    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(params_to_update, lr=lr, momentum=0.9,
                              weight_decay=weight_decay)
    else:
        print("Invalid optimizer name, exiting...")
        exit()

    return optimizer, lr, weight_decay


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement.
    # Each of these variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    # original 'swin_transformer'
    elif model_name == 'swin_transformer_base_224':
        model_ft = timm.create_model(
            'swin_base_patch4_window7_224_in22k', pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'swin_transformer_large_384':
        model_ft = timm.create_model(
            'swin_large_patch4_window12_384_in22k', pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        input_size = 384

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        log.message_loging(f"Epoch {epoch}/{num_epochs - 1}")
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        log.message_loging('-' * 10)
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double(
            ) / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            log.message_loging(
                f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()
        log.message_loging("")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    log.message_loging(
        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    print('Best val Acc: {:4f}'.format(best_acc))
    log.message_loging(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == '__main__':
    # Create model and log directory
    timestamp = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    result_dir = os.path.join('..', 'model', timestamp)
    create_dir(result_dir)

    # Create log file and start loging
    log = Log(result_dir)
    log.start_loging()

    log.message_loging(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch Version: {torch.__version__}")

    log.message_loging(f"Torchvision Version: {torchvision.__version__}")
    print(f"Torchvision Version: {torchvision.__version__}")

    # Assign the train and validation data directory
    data_dir = os.path.join("..", "data")

    # Models to choose from [resnet, inception, swin_transformer_base_224, swin_transformer_large_384]
    model_name = "swin_transformer_large_384"
    log.message_loging(f"Train model name: {model_name}")
    print(f"Train model name: {model_name}")

    # Number of classes in the dataset
    num_classes = 200
    log.message_loging(f"Number of predict classes: {num_classes}")
    print(f"Number of predict classes: {num_classes}")

    # Batch size for training (change depending on how much memory you have)
    batch_size = 8
    log.message_loging(f"Train batch size: {batch_size}")
    print(f"Train batch size: {batch_size}")

    # Number of epochs to train for
    num_epochs = 10
    log.message_loging(f"Train epochs: {num_epochs}")
    print(f"Train epochs: {num_epochs}")

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True
    log.message_loging(
        f"Feature extract: {'True' if feature_extract else 'False'}")
    print(f"Feature extract: {'True' if feature_extract else 'False'}")

    # Initialize the model for this run
    model_ft, input_size = initialize_model(
        model_name, num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    # print(model_ft)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomRotation(degrees=(-45, 45)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(
        data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.message_loging(f"Device: {device}")
    print(f"Device: {device}")

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    log.message_loging("Params to learn:")
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                log.message_loging(f"\t {name}")
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                log.message_loging(f"\t {name}")
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_name = 'SGD'
    optimizer_ft, lr, weight_decay = initialize_optimizer(
        optimizer_name, lr=1e-3)
    log.message_loging(
        f"Optimizer: {optimizer_name}, lr: {lr:.1e}, weight_decay: {weight_decay:.1e}")
    print(
        f"Optimizer: {optimizer_name}, lr: {lr:.1e}, weight_decay: {weight_decay:.1e}")

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    log.message_loging(f"Loss function: {'Cross Entropy'}")
    print(f"Loss function: {'Cross Entropy'}")

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,
                                 num_epochs=num_epochs, is_inception=(model_name == "inception"))

    model_path = os.path.join(result_dir, f'{model_name}_{timestamp}.pkl')
    log.message_loging(f"Model store path: {model_path}")
    print(f"Model store path: {model_path}")
    torch.save(model_ft, model_path)
    # Plot the training curves of validation accuracy vs. number
    #  of training epochs for the transfer learning method and
    #  the model trained from scratch
    ohist = []

    ohist = [h.cpu().numpy() for h in hist]

    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, num_epochs+1), ohist, label="Pretrained")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    fig_path = os.path.join(result_dir, f'{model_name}_{timestamp}.png')
    plt.savefig(fig_path)
    plt.close('all')

    log.end_loging()
