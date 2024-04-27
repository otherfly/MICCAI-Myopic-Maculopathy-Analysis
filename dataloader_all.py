import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data_aug.gaussian_blur import GaussianBlur

class MyopicMaculopathyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform1, transform2, transform3):
        """
        csv_file (string): CSV 文件路径。
        root_dir (string): 图像文件夹路径。
        transform (callable, optional): 可选的转换操作，应用于样本。
        """
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.labels_frame.iloc[idx, 1]

        image1 = self.transform1(image)
        image2 = self.transform2(image)
        image3 = self.transform3(image)

        return image1, image2, image3, label

def load_data(csv_file, root_dir, batch_size=16):  # Updated image size for Swin Transformer

    data_transforms_swin_trans = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop(224),  # Swin Transformer standard input size
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # data_transforms_swin_trans = transforms.Compose([
    #     transforms.Resize((224,224)),

    #     # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    data_transforms_ae = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

    color_jitter = transforms.ColorJitter(0.8 , 0.8 , 0.8 , 0.2 )
    data_transforms_simclr = transforms.Compose([transforms.Resize((256,256)),
                                              transforms.RandomResizedCrop(size=224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * 224)),
                                              transforms.ToTensor()])
    # data_transforms_simclr = transforms.Compose([transforms.Resize((224,224)),
                                            
    #                                         #   GaussianBlur(kernel_size=int(0.1 * 224)),
    #                                           transforms.ToTensor()])

    dataset = MyopicMaculopathyDataset(csv_file=csv_file, root_dir=root_dir, transform1=data_transforms_swin_trans, transform2=data_transforms_ae, transform3=data_transforms_simclr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,drop_last=False)
    # dataloader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=True, num_workers=4,drop_last=True)
    return dataloader,dataset.__len__()


# 路径
train_csv = "/mnt/sda1/home/kailexiangzi/SimCLR/data/train_data/2. Groundtruths/1. MMAC2023_Myopic_Maculopathy_Classification_Training_Labels.csv"
train_root_dir = "/mnt/sda1/home/kailexiangzi/SimCLR/data/train_data/1. Images/1. Training Set/"

valid_csv = "/mnt/sda1/home/kailexiangzi/SimCLR/data/train_data/2. Groundtruths/2. MMAC2023_Myopic_Maculopathy_Classification_Validation_Labels.csv"
valid_root_dir = "/mnt/sda1/home/kailexiangzi/SimCLR/data/train_data/1. Images/2. Validation Set/"

# 加载数据
train_loader,_ = load_data(train_csv, train_root_dir)
valid_loader,valid_loader_len = load_data(valid_csv, valid_root_dir)
