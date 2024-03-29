import os
from os.path import join, split
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from glob import glob
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, folder, label_csv, args, verbosity = 1, num_classes = 10, transform = None, train_transform = None, stop_after = None):
        self.images = []
        self.labels = []
        self.label_idxs = []
        self.train_transform = train_transform

        image_paths = glob(join(folder, '*.png'))
        image_count = len(image_paths)

        if verbosity >= 2: print
        with open(label_csv, 'r') as fi:        
            for i, image_path in enumerate(image_paths):
                if verbosity >= 2: print '\t%d/%d\r' % (i + 1, image_count),
                image_name = split(image_path)[-1]

                with Image.open(image_path) as im:
                    if transform:
                        im = transform(im)
                    im = np.array(im)                    
                    self.images.append(im)

                label_idx = int(fi.readline().split(',')[-1])
                if args.class_output:
                    label = [0] * (num_classes + 1)
                else:
                    label = [0] * num_classes
                    
                label[label_idx] = 1
                label = torch.FloatTensor(label)
                self.labels.append(label)
                self.label_idxs.append(label_idx)

                if stop_after and i == stop_after - 1:
                    break
        if verbosity >= 2: print

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = self.images[idx]
        label = self.labels[idx]
        label_idx = self.label_idxs[idx]

        if self.train_transform:
            im = Image.fromarray(im)
            im = self.train_transform(im)

        t = transforms.ToTensor()
        im_tensor = t(im)
        t = transforms.Normalize([0], [1])
        im_tensor = t(im_tensor)
        im.close()

        data = {
            'image' : im_tensor,
            'label' : label,
            'idx' : label_idx
        }

        return data
        



