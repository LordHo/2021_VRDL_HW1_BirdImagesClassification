import os
import time
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class_dict = {}
train_class = {}


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def load_classes(classes_file_path):
    classes_file = open(classes_file_path, 'r')
    for line in classes_file.readlines():
        class_num, class_name = line.rstrip('\n').split('.')
        class_dict[class_name] = class_num


def load_tain_class():
    species_names = os.listdir(os.path.join('..', 'data', 'train'))
    for i, specie_name in enumerate(species_names):
        train_class[i+1] = specie_name


class EvalDataset(Dataset):
    def __init__(self, image_order_path, image_folder, input_size):
        self.image_order_path = image_order_path
        self.image_folder = image_folder
        self.input_size = input_size

        # Load the eval image order from file
        self.images = [name.rstrip('\n') for name in open(
            self.image_order_path, 'r').readlines()]

        shape = (self.input_size, self.input_size)
        self.transforms = transforms.Compose([
            transforms.Resize(shape),
            transforms.CenterCrop(self.input_size),
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


def load_model(model_path, model_name):
    input_size = 0

    # Select the input size of corresponding model
    if model_name == "resnet":
        """ Resnet50
        """
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        """
        input_size = 299

    elif model_name == 'swin_transformer_base_224':
        """ Swin Transformer with image size 224x224
        """
        input_size = 224

    elif model_name == 'swin_transformer_large_384':
        """ Swin Transformer with image size 384x384
        """
        input_size = 384

    else:
        print("Invalid model name, exiting...")
        exit()

    # Load the corresponding model weights
    model_ft = torch.load(model_path)
    return model_ft, input_size


def eval(model, dataloader, result_path):
    # Open the file that store the eval result
    result = open(result_path, 'w')

    # Change the model mode to eval mode
    model.eval()

    # To eval each image according to eval image order
    for index, (inputs, inputs_name) in enumerate(tqdm(dataloader)):
        # Send input image to same device of model
        inputs = inputs.to(device)

        # eval the image
        outputs = model(inputs)

        # Take the hightest class number as eval result
        _, preds = torch.max(outputs, 1)

        # Convert the eval result number to final class name and class number
        name = train_class[int(preds[0])+1]
        res = class_dict[name]

        # Record the eval reult to file
        if (index + 1) == len(dataloader):
            result.write(f'{inputs_name[0]} {res}.{name}')
        else:
            result.write(f'{inputs_name[0]} {res}.{name}\n')

    result.close()


if __name__ == '__main__':
    # Create a timestamp for each eval result
    timestamp = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())

    """ Load trained model
        Change the model name and timestamp to the model you want to eval.
    """
    model_name = 'swin_transformer_large_384'
    timestamp = '2021-10-30 23-41-54'
    model_path = os.path.join('..', 'model', f'{model_name}_{timestamp}.pkl')
    model, input_size = load_model(model_path, model_name)

    # Create evaluation dataset
    eval_image_order_path = os.path.join('..', 'data', 'testing_img_order.txt')
    eval_image_dir = os.path.join('..', 'data', 'test')
    eval_dataset = EvalDataset(
        eval_image_order_path, eval_image_dir, input_size=input_size)

    # Create evaluation dataloader
    eval_dataloader = DataLoader(eval_dataset, batch_size=1)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model = model.to(device)

    """
        Load the classification classes names and numbers 
            and store as dictionary, such as {class name: class number}.
    """
    classes_file_path = os.path.join('..', 'data', 'classes.txt')
    load_classes(classes_file_path)

    """
        Load train classes names and numbers and store as dictionary,
        such as {class number: class name}.
        This class number is different as above. 
        This class number is according to the order of 
            the class names in window directory order.
    """
    load_tain_class()

    # The directory of the evaluation result stored
    result_dir = os.path.join(
        '..', 'result', f'eval_{model_name}_{timestamp}')
    # Create the above directory
    create_dir(result_dir)

    # the path of the evaluation result stored
    result_path = os.path.join(
        result_dir, f'eval_{model_name}_{timestamp}.txt')

    eval(model, eval_dataloader, result_path)
