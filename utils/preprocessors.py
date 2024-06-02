import os

import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image

BATCH_SIZE = 32


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
    random_rotation_degrees = 15

    if model_type == "Scratch":
        resize_param = 144
        center_crop_param = 128

    elif model_type == "Pretrained":
        resize_param = 256
        center_crop_param = 224
    else:
        raise ValueError("Unknown model type: must be 'Scratch' or 'Pretrained'")

    # Transform Pipeline for training data (includes Data Augmentation)
    train_transform = v2.Compose([
        v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        v2.RandomHorizontalFlip(p=0.5),  # Data augmentation
        v2.RandomRotation(degrees=random_rotation_degrees),  # Data augmentation
        v2.Resize(resize_param, antialias=True),  # Resize
        v2.CenterCrop(center_crop_param),  # Center crop
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])  # Normalize
    ])

    # Transform Pipeline for val/test data (no Data Augmentation)
    val_test_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(resize_param, antialias=True),
        v2.CenterCrop(center_crop_param),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
    ])

    # train_transform = v2.Compose([
    #     v2.Resize(resize_param),
    #     v2.RandomHorizontalFlip(),
    #     # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #     v2.RandomRotation(degrees=random_rotation_degrees),
    #     v2.CenterCrop(center_crop_param),
    #     v2.ToTensor(),
    #     v2.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
    # ])
    #
    # val_test_transform = transforms.Compose([
    #     transforms.Resize(resize_param),
    #     transforms.CenterCrop(center_crop_param),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
    # ])

    return train_transform, val_test_transform


def preprocess(model_type, data_dir="./data"):
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

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    benchmark_loader = DataLoader(test_data, batch_size=16, shuffle=False)  # collate_fn=my_collate)

    return train_loader, val_loader, benchmark_loader
