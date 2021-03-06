import argparse

parser = argparse.ArgumentParser()
parser.add_argument('out_folder', type = str)
parser.add_argument('-b', '--batch_size', type = int, default = 64)
parser.add_argument('-n', '--noise_dim', type = int, default = 100)
parser.add_argument('-e', '--epochs', type = int, default = 5000)
parser.add_argument('-r', '--resume', type = str, default = None)
parser.add_argument('-v', '--verbosity', type = int, default = 1)
parser.add_argument('-d', '--dual_discrim', action = 'store_true')
parser.add_argument('-c', '--class_output', action = 'store_true')
parser.add_argument('-q', '--quick_load', action = 'store_true')

args = parser.parse_args()

if args.verbosity >= 1: print 'Loading imports... ',
from random import randint
import os
from os.path import join, isdir
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from BasicDiscriminator import BasicDiscriminator
from DoubleDiscriminator import DoubleDiscriminator
from BasicGenerator import BasicGenerator
from ImageDataset import ImageDataset
from support import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import shutil

if not isdir(args.out_folder):
    os.mkdir(args.out_folder)

if args.verbosity >= 1: print 'Done!' 

if args.verbosity >= 1: print 'Loading datasets... ',

t_load = transforms.Compose([
    transforms.Pad(4)    
])

t_train = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.Resize((64,64))    
])

dataset_folder = '/scratch/bbeyers/CSC249_project/mnist/training'
dataset_csv = '/scratch/bbeyers/CSC249_project/mnist/training.csv'

if args.quick_load:
    train_dataset = ImageDataset(dataset_folder, dataset_csv, args, transform = t_load, train_transform = t_train, stop_after = args.batch_size)
else:
    train_dataset = ImageDataset(dataset_folder, dataset_csv, args, transform = t_load, train_transform = t_train)

train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4)
if args.verbosity >= 1: print 'Done!'


if args.verbosity >= 1: print 'Initializing networks... ',

if args.dual_discrim:
    generator = BasicGenerator(z_dim = 100, use_bias = True, dual_input = True)
    discriminator = BasicDiscriminator(1, use_bias = True, dual_output=True)
if args.class_output:
    generator = BasicGenerator(z_dim = 100, use_bias = True, dual_input = True)
    discriminator = BasicDiscriminator(11, use_bias = True)
else:
    generator = BasicGenerator(z_dim = 100, use_bias = True)
    discriminator = BasicDiscriminator(1, use_bias = True)

generator.apply(weights_init)
discriminator.apply(weights_init)

start_epoch = 0
best_G_loss = 1000

g_optimizer = optim.Adam(generator.parameters(), lr = 1e-4, betas = (.5, .999))
d_optimizer = optim.Adam(discriminator.parameters(), lr = 1e-4, betas = (.5, .999))

if args.resume:
    if os.path.isfile(args.resume):
        print("Resuming training from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        best_G_loss = checkpoint['best_G_loss']
        discriminator.load_state_dict(checkpoint['D_state_dict'])
        generator.load_state_dict(checkpoint['G_state_dict'])
        d_optimizer.load_state_dict(checkpoint['D_opt'])
        g_optimizer.load_state_dict(checkpoint['G_opt'])
        print("Success! Training from epoch {}."
                .format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

if torch.cuda.device_count() > 1:
    print "Using %d gpus." % torch.cuda.device_count()
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()

if args.verbosity >= 1: print 'Done!'

if args.dual_discrim:
    class_loss_fn = nn.MultiLabelSoftMarginLoss()
loss_fn = nn.BCELoss()

D_total_fake_loss = 0
D_total_fake_binary_loss = 0
D_total_fake_class_loss = 0

D_total_real_loss = 0
D_total_real_binary_loss = 0
D_total_real_class_loss = 0

G_total_loss = 0
G_total_binary_loss = 0
G_total_class_loss = 0

for epoch in xrange(0):
    if args.resume: break
    for i, data in enumerate(train_loader):
        D_loss = train_D(
            discriminator,
            generator,
            d_optimizer,
            data,
            args
        )

        if args.dual_discrim:
            D_real_binary_loss, D_real_class_loss, D_fake_binary_loss, D_fake_class_loss = D_loss
            D_real_loss = D_real_binary_loss + D_real_class_loss

            D_total_real_binary_loss += D_real_binary_loss.data[0]
            D_total_real_class_loss += D_real_class_loss.data[0]

            D_total_fake_binary_loss += D_fake_binary_loss.data[0]
            D_total_fake_class_loss += D_fake_class_loss.data[0]
        else:
            D_real_loss, D_fake_loss = D_loss

            D_total_fake_loss += D_fake_loss.data[0]
            D_total_real_loss += D_real_loss.data[0]

        if i % 100 == 0:
            if args.dual_discrim:
                s = '[init %d %d] D real loss: %.3E, %.3E D fake loss: %.3E, %.3E'
                f = (epoch, i, D_total_real_binary_loss / 100, D_total_real_class_loss / 100, D_total_fake_binary_loss / 100, D_total_fake_class_loss / 100)                
            else:
                s = '[init %d %d] D real loss: %.3E D fake loss: %.3E'
                f = (epoch, i, D_total_real_loss / 100, D_total_fake_loss / 100)
            print s % f
            
            D_total_fake_loss = 0
            D_total_fake_binary_loss = 0
            D_total_fake_class_loss = 0

            D_total_real_loss = 0
            D_total_real_binary_loss = 0
            D_total_real_class_loss = 0

for epoch in range(start_epoch, start_epoch + args.epochs):    
    for i, data in enumerate(train_loader):
        D_loss = train_D(
            discriminator,
            generator,
            d_optimizer,
            data,
            args
        )

        if args.dual_discrim:
            D_real_binary_loss, D_real_class_loss, D_fake_binary_loss, D_fake_class_loss = D_loss

            D_total_real_binary_loss += D_real_binary_loss.data[0]
            D_total_real_class_loss += D_real_class_loss.data[0]

            D_total_fake_binary_loss += D_fake_binary_loss.data[0]
            D_total_fake_class_loss += D_fake_class_loss.data[0]
        else:
            D_real_loss, D_fake_loss = D_loss
            
            D_total_fake_loss += D_fake_loss.data[0]
            D_total_real_loss += D_real_loss.data[0]

        # G Training
        G_loss = train_G(
            discriminator,
            generator,
            g_optimizer,
            args
        )

        if args.dual_discrim:
            G_binary_loss, G_class_loss = G_loss

            G_total_binary_loss += G_binary_loss.data[0]
            G_total_class_loss += G_class_loss.data[0]
        else:
            G_total_loss += G_loss.data[0]        

        if i % 100 == 0:
            if args.dual_discrim:
                s = '[%d %d] D real loss: %.3E, %.3E D fake loss: %.3E, %.3E G loss: %.3E, %.3E'
                f = (
                    epoch,
                    i,
                    D_total_real_binary_loss / 100,
                    D_total_real_class_loss / 100,
                    D_total_fake_binary_loss / 100,
                    D_total_fake_class_loss / 100,
                    G_total_binary_loss / 100,
                    G_total_class_loss / 100
                )
                
            else:
                s = '[%d %d] D real loss: %.3E D fake loss: %.3E G loss: %.3E'
                f = (epoch, i, D_total_real_loss / 100, D_total_fake_loss / 100, G_total_loss / 100)
            print s % f
            
            D_total_fake_loss = 0
            D_total_fake_binary_loss = 0
            D_total_fake_class_loss = 0

            D_total_real_loss = 0
            D_total_real_binary_loss = 0
            D_total_real_class_loss = 0

            G_total_loss = 0
            G_total_binary_loss = 0
            G_total_class_loss = 0

    if epoch % 1 == 0:        
        save_samples(
            generator,
            args.out_folder,
            'samples_epoch_%d.png' % epoch,
            args
        )
    
    if epoch % 5 == 0:
        G_avg_loss = G_total_loss / len(train_loader)

        torch.save(
            {
                'epoch': epoch + 1,
                'D_state_dict': discriminator.state_dict(),
                'G_state_dict': generator.state_dict(),
                'best_G_loss': best_G_loss,
                'D_opt' : d_optimizer.state_dict(),
                'G_opt' : g_optimizer.state_dict(),
            },
            join(args.out_folder, '%d.pth' % epoch)
        )
        
        
