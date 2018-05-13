import os
import re
from os.path import join, split
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from glob import glob
from PIL import Image, ImageOps

class ArtDataset(Dataset):
    def __init__(self, args, transform = None, train_transform = None, stop_after = None):
        self.images = []
        self.train_transform = train_transform

        verbosity = args.verbosity

        image_paths = glob(join('/public', 'bbeyers', 'CSC249_project', 'kaggle_128', '*.jpg')) + \
                      glob(join('/public', 'bbeyers', 'CSC249_project', 'wikiart_128', '*.png'))
        image_count = len(image_paths)
        print image_count

        # if verbosity >= 2: print        
        # for i, image_path in enumerate(image_paths):
        #     if verbosity >= 2: print '\t%d/%d\r' % (i + 1, image_count),
        #     image_name = split(image_path)[-1]
        #     try:
        #         with Image.open(image_path) as im:
        #             w, h = im.size
        #             if w < 64 or h < 64: continue
        #             if transform:
        #                 im = transform(im)                    
        #             self.images.append(np.array(im))
        #     except Exception:
        #         if verbosity >= 1: print 'Failed on %s' % image_path

        #     if stop_after and len(self.images) >= stop_after:
        #         break
        # if verbosity >= 2: print

        self.images = image_paths

        #regex = re.compile(r'.*(w|k)_[0-9]*\.png')
        #self.filenames = filter(regex.match,image_paths)
        self.filenames = [path.split('/')[5] for path in image_paths]
        self.is_stored = [False]*len(self.filenames)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # im = self.images[idx]
        # im = Image.fromarray(im)
        im = Image.open(self.images[idx])
        fname = self.filenames[idx]
        stored = self.is_stored[idx]
        im = im.convert('RGB')
        w, h = im.size
        pad = [0,0,0,0]
        if w < 64:
            dw = (64 - w)
            pad[0] = dw / 2
            pad[2] = dw - (dw / 2)
        if h < 64:
            dh = (64 - h)
            pad[1] = dh / 2
            pad[3] = dh - (dh / 2)

        if w < 64 or h < 64:
            im = ImageOps.expand(im, tuple(pad))        

        if self.train_transform:
            im = self.train_transform(im)

        t = transforms.ToTensor()
        im_tensor = t(im)
        t = transforms.Normalize([0], [1])
        im_tensor = t(im_tensor)
        im.close()        


        data = {
            'image' : im_tensor,
            'filename' : fname,
            'is_stored': stored
        }

        return data
        
