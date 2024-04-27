import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MyopicMaculopathyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file (string): CSV 文件路径。
        root_dir (string): 图像文件夹路径。
        transform (callable, optional): 可选的转换操作，应用于样本。
        """
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.labels_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

def load_data(csv_file, root_dir, batch_size=32, img_size=(256, 256)):  # Updated image size for Swin Transformer
    data_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(224),  # Swin Transformer standard input size
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = MyopicMaculopathyDataset(csv_file=csv_file, root_dir=root_dir, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader


# 路径
train_csv = 'D:/MICCAI MMAC 2023/1. Classification of Myopic Maculopathy_train/2. Groundtruths/1. MMAC2023_Myopic_Maculopathy_Classification_Training_Labels.csv'
train_root_dir = 'D:/MICCAI MMAC 2023/1. Classification of Myopic Maculopathy_train/1. Images/1. Training Set'

valid_csv = 'D:/MICCAI MMAC 2023/1. Classification of Myopic Maculopathy_valid/2. Groundtruths/2. MMAC2023_Myopic_Maculopathy_Classification_Validation_Labels.csv'
valid_root_dir = 'D:/MICCAI MMAC 2023/1. Classification of Myopic Maculopathy_valid/1. Images/2. Validation Set'

# 加载数据
train_loader = load_data(train_csv, train_root_dir)
valid_loader = load_data(valid_csv, valid_root_dir)
