import os
import pickle

import torch
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
from miscc.utils import get_filenames, get_filenames_from_pickle

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, img_data_dir, filename_path, transform=None, max_num=30000):

        self.max_num = max_num
        self.transform = transform
        if filename_path is not None and os.path.splitext(filename_path)[-1] == '.pickle':
            self.filenames = get_filenames_from_pickle(img_data_dir, filename_path)
        else:
            self.filenames = get_filenames(img_data_dir)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.filenames)

    def __getitem__(self, index):
        'Generates one sample of data'
        img = Image.open(self.filenames[index]).convert('RGB')
        # Convert image and label to torch tensors
        if self.transform is not None:
            img = self.transform(img)
        return img

if __name__ == '__main__':
    path = "/media/twilightsnow/workspace/gan/AttnGAN/output/birds_attn2_2018_06_24_14_52_20/Model/netG_avg_epoch_300"
    batch_size = 16
    dataset = Dataset(path, transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    print(dataset.__len__())
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    for i, batch in enumerate(dataloader):
        print(batch)
        break