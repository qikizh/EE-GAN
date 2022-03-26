import os
import pickle

import torch
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, path, pickle_path=None, transform=None, max_num=30000):
        'Initialization'

        self.max_num = max_num
        if pickle_path is None:
            self.file_names = self.get_filenames(path, max_num)
        else:
            self.file_names = self.get_filenames_from_pickle(path, pickle_path)

        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_names)

    def __getitem__(self, index):
        'Generates one sample of data'
        img = Image.open(self.file_names[index]).convert('RGB')
        # Convert image and label to torch tensors
        if self.transform is not None:
            img = self.transform(img)
        return img

    @staticmethod
    def get_filenames(data_path, max_num):
        images = []
        for path, subdirs, files in os.walk(data_path):
            for name in files:
                if name.rfind('jpg') != -1 or name.rfind('png') != -1:
                    filename = os.path.join(path, name)
                    if os.path.isfile(filename):
                        images.append(filename)

        if len(images) > max_num:
            images = images[:max_num]

        return images

    @staticmethod
    def get_filenames_from_pickle(data_dir, pickle_path):
        with open(pickle_path, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (pickle_path, len(filenames)))
        # filenames

        image_dir = os.path.join(data_dir, "images")

        for ix in range(len(filenames)):
            key = filenames[ix]
            image_path = os.path.join(image_dir, "%s.jpg" % key)
            filenames[ix] = image_path

        return filenames

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