import os

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image

# Define the data loaders

batch_size = 32
data_dir = 'data'


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA')


class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends torchvision.datasets.ImageFolder"""

    def __getitem__(self, index):
        # This is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # The image file path
        path = self.imgs[index][0]
        # Make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def get_transformer(model_type):
    train_transform = None
    val_test_transform = None

    if model_type == "Scratch":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize(144),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
        ])
    elif model_type == "Pretrained":
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize(144),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
        ])

    return train_transform, val_test_transform


def preprocess(model_type):
    train_transform, val_test_transform = get_transformer(model_type=model_type)

    train_data = ImageFolder(root=os.path.join(data_dir, 'train'),
                             transform=train_transform,
                             loader=pil_loader)
    val_data = ImageFolder(root=os.path.join(data_dir, 'val'),
                           transform=val_test_transform,
                           loader=pil_loader)
    test_data = ImageFolderWithPaths(root=os.path.join(data_dir, 'test'),
                                     transform=val_test_transform,
                                     loader=pil_loader)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    benchmark_loader = DataLoader(test_data, batch_size=16, shuffle=False)  # collate_fn=my_collate)

    return train_loader, val_loader, benchmark_loader
