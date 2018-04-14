import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type = int, default = 64)
parser.add_argument('-n', '--noise_dim', type = int, default = 100)
parser.add_argument('-e', '--epochs', type = int, default = 5000)
parser.add_argument('-r', '--resume', type = str, default = None)
parser.add_argument('-v', '--verbosity', type = int, default = 1)
parser.add_argument('-d', '--dual_discrim', action = 'store_true')
parser.add_argument('-q', '--quick_load', action = 'store_true')

args = parser.parse_args()

if args.v >= 1: print 'Loading imports... ',
from random import randint
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from BasicDiscriminator import BasicDiscriminator
from BasicGenerator import BasicGenerator
from ImageDataset import ImageDataset
from support import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import shutil

if args.v >= 1: print 'Done!' 

if args.v >= 1: print 'Loading datasets... ',

t_load = transforms.Compose([
    transforms.Pad(4)    
])

t_train = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.Resize((64,64))    
])

dataset_folder = '/home/bbeyers/Documents/CSC249_project/mnist/training'
dataset_csv = '/home/bbeyers/Documents/CSC249_project/mnist/training.csv'

if args.quick_load:
    train_dataset = ImageDataset(dataset_folder, dataset_csv, transform = t_load, train_transform = t_train, stop_after=args.batch_size)
else:
    train_dataset = ImageDataset(dataset_folder, dataset_csv, transform = t_load, train_transform = t_train)

train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 2)
if args.v >= 1: print 'Done!'

generator = BasicGenerator(z_dim = 100, use_bias = True)
g_optimizer = optim.Adam(generator.parameters(), lr = 2e-4, betas = (.5, .999))

if args.dual_discrim:
    discriminator = BasicDiscriminator(use_bias = True, dual_output = True)
else:
    discriminator = BasicDiscriminator(use_bias = True)

generator.apply(weights_init)
discriminator.apply(weights_init)

start_epoch = 0
best_G_loss = 1000

d_optimizer = optim.Adam(discriminator.parameters(), lr = 2e-4, betas = (.5, .999))

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        best_G_loss = checkpoint['best_G_loss']
        discriminator.load_state_dict(checkpoint['D_state_dict'])
        generator.load_state_dict(checkpoint['G_state_dict'])
        d_optimizer.load_state_dict(checkpoint['D_opt'])
        g_optimizer.load_state_dict(checkpoint['G_opt'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "gpus.")
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()

if args.dual_discrim:
    class_loss_fn = nn.MultiLabelSoftMarginLoss()
loss_fn = nn.BCELoss()

D_total_fake_loss = 0
D_total_real_loss = 0
G_total_loss = 0

for epoch in xrange(2):
    if args.resume: break
    for i, data in enumerate(train_loader):
        # D Real Training
        D_real_loss, D_fake_loss = train_D(
            discriminator,
            generator,
            d_optimizer,
            data,
            args.noise_dim,
            torch.cuda.is_available(),
            args.dual_discrim,
            args.batch_size,
            train_fake = True
        )

        D_total_fake_loss += D_fake_loss.data[0]
        D_total_real_loss += D_real_loss.data[0]

        if i % 100 == 0:            
            if args.dual_discrim:
                print('[init %d %d] D real loss: %.8f D fake loss: %.8f D acc: %.5f' % \
                    (epoch, i, D_total_real_loss / 100, D_total_fake_loss / 100, acc))
            else:
                print('[init %d %d] D real loss: %.8f D fake loss: %.8f' % \
                    (epoch, i, D_total_real_loss / 100, D_total_fake_loss / 100))
            D_total_fake_loss = 0
            D_total_real_loss = 0

for epoch in range(start_epoch, start_epoch + args.epochs):    
    for i, data in enumerate(train_loader):
        D_real_loss, D_fake_loss = train_D(
            discriminator,
            generator,
            d_optimizer,
            data,
            args.noise_dim,
            torch.cuda.is_available(),
            args.dual_discrim,
            args.batch_size
        )

        D_total_fake_loss += D_fake_loss.data[0]
        D_total_real_loss += D_real_loss.data[0]

        if i % 1 == 0:
            # G Training
            G_loss = train_G(
                discriminator,
                generator,
                g_optimizer,
                args.noise_dim,
                torch.cuda.is_available(),
                args.dual_discrim,
                args.batch_size
            )

            G_total_loss += G_loss.data[0]

        if i % 100 == 0:            
            print('[%d %d] D real loss: %.8f D fake loss: %.8f G loss: %.8f' % \
                (epoch, i, D_total_real_loss / 100, D_total_fake_loss / 100, G_loss.data[0]))
            D_total_fake_loss = 0
            D_total_real_loss = 0
            G_total_loss = 0

    if epoch % 1 == 0:
        noise = torch.FloatTensor(args.batch_size, args.noise_dim, 1, 1).normal_(0, 1)
        if args.dual_discrim:
            class_label = gen_class_label(args.batch_size)
            noise = torch.cat([noise, class_label], 1)
        noise = Variable(noise)
    
        if torch.cuda.is_available():
            noise = noise.cuda()
        fake_images = generator(noise)    
        utils.save_image(fake_images.data, '/home/bbeyers/Documents/CSC249_project/%s/fake_samples_epoch_%03d.png' % ('long_binary', epoch))
    
    if epoch % 5 == 0:
        G_avg_loss = G_total_loss / len(train_loader)
        if G_avg_loss < best_G_loss:
            save_checkpoint({
            'epoch': epoch + 1,
            'D_state_dict': discriminator.state_dict(),
            'G_state_dict': generator.state_dict(),
            'best_G_loss': best_G_loss,
            'D_opt' : d_optimizer.state_dict(),
            'G_opt' : g_optimizer.state_dict(),
            },
            True,
            '/home/bbeyers/Documents/CSC249_project/long_binary/%d.pth' % epoch            
            )
        else:
            save_checkpoint({
            'epoch': epoch + 1,
            'D_state_dict': discriminator.state_dict(),
            'G_state_dict': generator.state_dict(),
            'best_G_loss': best_G_loss,
            'D_opt' : d_optimizer.state_dict(),
            'G_opt' : g_optimizer.state_dict(),
            },
            False,
            '/home/bbeyers/Documents/CSC249_project/long_binary/%d.pth' % epoch            
            )
        
        
