import argparse
import sys

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
parser.add_argument('-s', '--save', action = 'store_true')

args = parser.parse_args()

sys.path.append('/home/sloane/ArtBot/pytorch_gan/meta/')

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
from FeatureExtractor import FeatureExtractor
from BasicGenerator import BasicGenerator
from ArtDataset import ArtDataset
from support import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import shutil
import meta

#os.chdir(r'./meta/')
metafile = open('/home/sloane/ArtBot/pytorch_gan/meta/kaggle_meta.csv','r')
art_titles, d1, d2, d3 = meta.kaggle_analysis(fi = metafile)
art_titles, d1, d2, d3 = meta.wikiart_analysis(titles=art_titles)
#print art_titles.keys()[:20]
d1 = d2 = d3 = 0

if not isdir(args.out_folder):
    os.mkdir(args.out_folder)

if args.verbosity >= 1: print 'Done!' 

if args.verbosity >= 1: print 'Loading datasets... ',

t_train = transforms.Compose([
    transforms.RandomCrop(64)
])

if args.quick_load:
    train_dataset = ArtDataset(args, train_transform = t_train, stop_after = args.batch_size)
else:
    train_dataset = ArtDataset(args, train_transform = t_train)

train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 2)
if args.verbosity >= 1: print 'Done!'


if args.verbosity >= 1: print 'Initializing networks... ',

if args.verbosity >= 2: print 'G & D, ',
if args.dual_discrim:
    generator = BasicGenerator(z_dim = 100, use_bias = True, dual_input = True, output_channels = 3)
    discriminator = BasicDiscriminator(1, use_bias = True, dual_output=True, input_channels = 3)
if args.class_output:
    generator = BasicGenerator(z_dim = 100, use_bias = True, dual_input = True, output_channels = 3)
    discriminator = BasicDiscriminator(11, use_bias = True, input_channels = 3)
else:
    generator = BasicGenerator(z_dim = 100, use_bias = True, output_channels = 3)
    discriminator = BasicDiscriminator(1, use_bias = True, input_channels = 3)
if args.verbosity >= 2: print 'FE'
extractor = FeatureExtractor(11, use_bias = True, input_channels = 3)


if args.verbosity >= 2: print 'apply weights'
generator.apply(weights_init)
discriminator.apply(weights_init)
w = list(discriminator.parameters())
we = list(extractor.parameters())
#print [len(w), len(we)]
#for param in w:
#    print [type(param.data),param.size()]
we = w

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
        discriminator.load_state_dict(checkpoint['D_state_dict'], strict = False)
        generator.load_state_dict(checkpoint['G_state_dict'], strict = False)
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
    extractor = nn.DataParallel(extractor)

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    extractor.cuda()

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

if args.verbosity >= 1: print 'Start Epochs'

for epoch in xrange(0):
    if args.resume: break
    with open("/home/sloane/ArtBot/pytorch_gan/meta/prefixes.txt",'w+') as pref:
        pref.seek(0)
        pref.write("")
    with open("/home/sloane/ArtBot/pytorch_gan/meta/titles.txt",'w+') as titf:
        titf.seek(0)
        titf.write("")
    print "Epoch 1"
    for i, data in enumerate(train_loader):
        D_loss = train_D(
            discriminator,
            generator,
            extractor,
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
        print i,":: ",
        print "Train D, ",
        D_loss = train_D(
            discriminator,
            generator,
            extractor,
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

        # D Feature Extraction
        print "FeatureExtract, "
        image = Variable(data['image'])
        #title = Variable(data['title'])
        filenames = data['filename']
        stored = data['is_stored']
        #print filenames[:20]
        w = list(discriminator.parameters())
        we = list(extractor.parameters())
        we = w
        features = extractor(image)
        features = features.data
        #print(features.size())
        #print(len(image))
        titles = [art_titles[fname] for fname in filenames]
        prefixes = [''.join([str(round(x,2)*10)+"_" for x in vec[:,0,0]]) for vec in features]
        prefixes = [prefix[:-1] for prefix in prefixes]
        with open("/home/sloane/ArtBot/pytorch_gan/meta/prefixes.txt",'a') as prefixfile:
            for prefix,val in zip(prefixes,stored):
                if not val:
                    prefixfile.write("\n")
                    prefixfile.write(prefix)
        print prefixes
        with open("/home/sloane/ArtBot/pytorch_gan/meta/titles.txt",'a') as titlefile:
            for title,val in zip(titles,stored):
                if not val:
                    titlefile.write("\n")
                    titlefile.write(title)
                    val = True
        titles = [str(pre) + title for i,title,pre in zip(range(len(titles)),titles,prefixes)]
        print(titles)
        #print(discriminator(image))
        #print(features)

        # G Training
        print "Train G "
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

    if True: #args.save:
        print "Saving in %s on epoch %d" % (args.out_folder,epoch)
        save_samples(
            generator,
            extractor,
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
        
        
