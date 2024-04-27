from torch.utils.data import Dataset
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader

class ourDataset(Dataset):

    def __init__(self, path, start_index, end_index, transform=None):
        # Parameters
        self.path = path
        self.start_index = start_index
        self.end_index = end_index
        self.transform = transform
        # Preprocess
        self.name_list = os.listdir(self.path)

        
    def __len__(self):
        return (self.end_index - self.start_index)

    def __getitem__(self, index):
        # Read image
        index = self.start_index + index
        # Paths
        image_path = os.path.join(self.path, self.name_list[index])
        # Read
        slice = Image.open(image_path)

        if self.transform:
            # slice = Image.fromarray(slice)
            slice = self.transform(slice)
        return slice



def get_dataloaders(path, transform=None):
    image_path = path

    # Divide

    name_list = os.listdir(image_path)
    num_all = len(name_list)
    train_start, train_end = 0, num_all
  
    train_dataset = ourDataset(image_path, train_start, train_end, transform=transform)
     
    dataloaders = DataLoader(
        dataset=train_dataset,
        sampler=None,
        batch_size=32,
        # shuffle=shuffle,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )


    return dataloaders